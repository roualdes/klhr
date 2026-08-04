#pragma once

#include "bfgs.hpp"
#include "gausshermite.hpp"
#include "k_adaptation.hpp"
#include "klhr_numerics.hpp"
#include "normal_quantile.hpp"
#include "onlinepca.hpp"
#include "reflected_transport.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>
#include <welford.hpp>
#include <windowedadaptation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace klhr {

inline constexpr double positive_infinity =
  std::numeric_limits<double>::infinity();

struct KlhrOptions {
  std::uint64_t seed = 0;
  Eigen::Index N = 8;
  double tol = 1e-10;
  double grad_clip = positive_infinity;
  double sas_arg_clip = 30;
  double gtol = 1e-5;
  // Accept the Laplace line fit when its KL gradient residual (location
  // measured in proposal standard deviations) is below this. Tightening it
  // makes the KL optimizer run far more often: on funnel that roughly
  // doubled cost with no accuracy gain, and on an exactly Gaussian target
  // like ar1 the residual is ~0 so the setting has no effect at all.
  double laplace_kl_residual_tol = 1.0;
  std::size_t maxiter_bfgs = 32;
  std::size_t transport_maxiter_bfgs = 8;
  // Odd K only: an even K puts a self-transition atom at the middle rank,
  // which wastes a draw and is not part of the continuous kernel.
  std::size_t K = 15;
  bool adapt_K = true;
  std::size_t K_max = 63;
  std::size_t K_windowsize = 50;
  std::size_t K_windowscale = 2;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  Eigen::Index J = 1;
  double direction_lowrank_weight = 1.0;
  double direction_min_diag_fraction = 0.1;
  bool lowrank_during_warmup = true;
  double pca_freeze_fraction = 0.1;
  double transport_cov_shrink = 0.25;
  double transport_cov_ratio_cap = 4.0;
  double l = 0.0;
  std::size_t initial_transport_steps = 150;
  std::size_t transport_max_reflections = 500;
  double transport_initial_distance = 1.0;
  double transport_min_distance = 1e-8;
  double transport_max_distance = 1e6;
  double transport_max_logp_drop = 1000.0;
  double transport_max_segment_logp_drop = 500.0;
  double transport_max_endpoint_from_best_drop = 100.0;
  double transport_direction_persistence = 0.9;
  double transport_failure_direction_decay = 0.25;
  // Reflections allowed across the whole transport phase, as a multiple of
  // initial_transport_steps. Rare deep excursions stay affordable while a
  // target that wants max_reflections on every step gets throttled.
  std::size_t transport_reflection_budget_per_step = 75;
};

class BaseKLHR {
public:
  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;

  BaseKLHR(std::string stan_file, std::string json_file,
           const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(normalized_options_(options, bsm_.dim())),
    transport_(bsm_.dim(), transport_options_(opts_)),
    windowed_adaptation_(opts_.warmup, opts_.windowsize, opts_.windowscale),
    K_adaptation_(K_adaptation_config_(opts_, bsm_.dim())),
    online_moments_(bsm_.dim()),
    K_online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), opts_.J, opts_.l, opts_.tol),
    projected_moments_(opts_.J) {

    if (opts_.seed == 0) {
      std::random_device rd;
      std::uint64_t r1 = rd();
      std::uint64_t r2 = rd();
      opts_.seed = (r1 << 32) ^ r2;
      if (opts_.seed == 0) {
        opts_.seed = 1;
      }
      rng_.seed(opts_.seed);
    }

    std::uniform_int_distribution<unsigned int> uniform_uint;
    mcmcpp::bsrng bsrng = bsm_.make_rng(uniform_uint(rng_));
    theta_.resize(dim());
    theta_ = bsm_.param_initialize(bsrng);

    w_.resize(opts_.N);
    x_.resize(opts_.N);
    gauss_hermite(opts_.N, w_, x_);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);

    initialize_pca_schedule_();
    reset_adaptation_to_defaults_(false);
    nfev_ = 0;
    acceptance_rate_ = 0.0;
    Eigen::VectorXd initial_grad = Eigen::VectorXd::Zero(dim());
    bsm_.log_density_gradient_noe(theta_, log_density_, initial_grad);
    ++nfev_;

    initialize_K_window_metric_(false);

    transport_.initialize({theta_, initial_grad, log_density_, true},
                          rng_, std_normal_);
    draw_ = 0;
  }

  virtual ~BaseKLHR() = default;

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  std::size_t overrelaxation_K() const {
    return opts_.K;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      auto result = transport_.step(
        bsm_, rng_, std_uniform_, std_normal_);
      nfev_ += result.evaluations;
      theta_ = result.state.theta;
      log_density_ = result.state.log_density;
      // const double acceptance_delta = result.moved - acceptance_rate_;
      // acceptance_rate_ += acceptance_delta / draw_;
      if (draw_ <= opts_.warmup) {
        (void) windowed_adaptation_.window_closed(draw_);
      }
      if (draw_ == opts_.initial_transport_steps) {
        apply_transport_handoff_(transport_.finish(rng_, std_normal_));
      }
      return bsm_.param_constrain(theta_);
    }

    const std::size_t K_draw = post_transport_warmup_draw(
      draw_, opts_.warmup, opts_.initial_transport_steps);
    if (K_draw > 0 && K_adaptation_.enabled()) {
      opts_.K = K_adaptation_.K();
    }

    Eigen::VectorXd rho = random_direction();
    const KlStepDiagnostics diagnostics = kl_step_(rho);
    if (K_draw > 0 && K_adaptation_.enabled()) {
      adapt_K_warmup_(rho, diagnostics);
    }
    adapt_warmup_(theta_, draw_);
    return bsm_.param_constrain(theta_);
  }

  Eigen::VectorXd random_direction() {
    const Eigen::Index D = dim();
    const bool use_sampling_direction =
      (opts_.lowrank_during_warmup || draw_ > opts_.warmup) && lowrank_ready_;
    Eigen::VectorXd rho = use_sampling_direction ?
      direction_noise_() : mean_direction_noise_();

    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_rng_(D);
      norm = rho.norm();
    }
    // Exactly unit, not norm + tol: kl_step_ recovers the canonical line
    // origin as theta - ((theta - mean).rho) rho, and that point is only the
    // same from both ends of the line when rho.rho is one.
    if (std::isfinite(norm) && norm > 0.0) {
      rho /= norm;
    } else {
      rho = Eigen::VectorXd::Zero(D);
      rho(0) = 1.0;
    }

    return rho;
  }

protected:
  // Fit the line-conditional approximation, in coordinates relative to
  // `center` (a canonical point on the line).
  virtual Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                                    const Eigen::VectorXd& rho) = 0;

  // Ordered overrelaxation of `from` with respect to the fitted density.
  virtual double overrelaxed_proposal_(const Eigen::VectorXd& eta,
                                       const double from) = 0;

  // log q(t) under the fitted density. Only ratios of this appear in the
  // Hastings ratio, so an omitted normalising constant is fine as long as
  // it does not depend on t.
  virtual double log_line_density_(const double t,
                                   const Eigen::VectorXd& eta) const = 0;

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
    bool hessian_usable = false;
    bool success = false;
  };

  LineModeEstimate fit_line_mode_(const Eigen::VectorXd& center,
                                  const Eigen::VectorXd& rho) {
    Eigen::VectorXd mode_init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd target_grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + center, value, target_grad);
      value = -value;
      g(0) = -target_grad.dot(rho);
    };

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init, bfgs_options_());
    nfev_ += mode.nfev;

    LineModeEstimate out;
    if (mode.x.size() == 1 && std::isfinite(mode.x(0))) {
      out.mode = mode.x(0);
    }
    double inverse_hessian = std::numeric_limits<double>::quiet_NaN();
    if (mode.hess_inv.rows() == 1 && mode.hess_inv.cols() == 1) {
      inverse_hessian = mode.hess_inv(0, 0);
    }
    out.hessian_usable =
      std::isfinite(inverse_hessian) && inverse_hessian > 0.0;
    if (out.hessian_usable) {
      out.log_scale = 0.5 * std::log(inverse_hessian);
    }
    out.success = mode.success;
    return out;
  }

  bfgs::BfgsOptions bfgs_options_() const {
    return {.gtol = opts_.gtol,
            .xrtol = opts_.gtol,
            .maxiter_bfgs = opts_.maxiter_bfgs};
  }

  // The Laplace fit along the line is a cheap candidate for the KL-optimal
  // fit, but it is only a candidate: accepting it unconditionally pins the
  // shape parameters at their defaults (for SAS, zero skew) and never tests
  // whether it is near the KL optimum. Every path therefore spends one KL
  // evaluation validating it, and optimizes when the residual is too large.
  template <typename EvaluateKl, typename TransformParameters>
  Eigen::VectorXd fit_line_with_kl_fallback_(
      const Eigen::VectorXd& center,
      const Eigen::VectorXd& rho,
      const Eigen::Index parameter_count,
      EvaluateKl evaluate_kl,
      TransformParameters transform_parameters) {
    const LineModeEstimate mode = fit_line_mode_(center, rho);

    Eigen::VectorXd init = Eigen::VectorXd::Zero(parameter_count);
    init(0) = mode.mode;

    auto kl = [&](const Eigen::VectorXd& eta,
                  double& value, Eigen::VectorXd& grad) {
      evaluate_kl(eta, mode.log_scale, value, grad);
    };

    if (!mode.success || !mode.hessian_usable) {
      bfgs::BfgsResult fit = bfgs::bfgs(kl, init, bfgs_options_());
      nfev_ += fit.nfev * opts_.N;
      const Eigen::VectorXd raw =
        fit.x.size() == parameter_count && fit.x.allFinite() ? fit.x : init;
      return transform_parameters(raw, mode.log_scale);
    }

    double initial_value = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd initial_grad;
    kl(init, initial_value, initial_grad);
    nfev_ += opts_.N;

    const bool valid_initial =
      std::isfinite(initial_value) &&
      !numerics::is_bad_kl(initial_value) &&
      initial_grad.size() == parameter_count &&
      initial_grad.allFinite();
    if (valid_initial) {
      Eigen::VectorXd residual = initial_grad;
      // Express the location residual in proposal-standard-deviation units.
      residual(0) *= scale_from_log_(mode.log_scale);
      if (residual.lpNorm<Eigen::Infinity>() <=
          opts_.laplace_kl_residual_tol) {
        return transform_parameters(init, mode.log_scale);
      }
    }

    // Reuse the validation evaluation as BFGS's initial evaluation.
    bool initial_evaluation_available = true;
    auto cached_kl = [&](const Eigen::VectorXd& eta,
                         double& value, Eigen::VectorXd& grad) {
      const bool at_initial =
        eta.size() == init.size() &&
        (eta.array() == init.array()).all();
      if (initial_evaluation_available && at_initial) {
        initial_evaluation_available = false;
        value = initial_value;
        grad = initial_grad;
        return;
      }
      kl(eta, value, grad);
    };

    bfgs::BfgsResult fit = bfgs::bfgs(cached_kl, init, bfgs_options_());
    // The cached first evaluation was already charged above.
    nfev_ += (fit.nfev > 0 ? fit.nfev - 1 : 0) * opts_.N;
    const Eigen::VectorXd raw =
      fit.x.size() == parameter_count && fit.x.allFinite() ? fit.x : init;
    return transform_parameters(raw, mode.log_scale);
  }

  // K=1 is the independence kernel, identical to K=0. Even K adds an
  // unmodelled self-transition atom at the middle rank, so round down to the
  // nearest usable odd level.
  static std::size_t force_odd_K_(const std::size_t K) {
    if (K <= 1) {
      return 0;
    }
    return K % 2 == 0 ? K - 1 : K;
  }

  static KlhrOptions normalized_options_(KlhrOptions options,
                                         const Eigen::Index dim) {
    const Eigen::Index D = dim;
    options.N = std::max<Eigen::Index>(1, options.N);
    if (!(options.tol > 0.0) || !std::isfinite(options.tol)) {
      options.tol = 1e-10;
    }
    if (std::isfinite(options.grad_clip)) {
      options.grad_clip = std::abs(options.grad_clip);
    } else {
      options.grad_clip = positive_infinity;
    }
    if (!(options.sas_arg_clip > 0.0) ||
        !std::isfinite(options.sas_arg_clip)) {
      options.sas_arg_clip = 30.0;
    }
    if (!(options.gtol > 0.0) || !std::isfinite(options.gtol)) {
      options.gtol = 1e-3;
    }
    if (!(options.laplace_kl_residual_tol >= 0.0) ||
        !std::isfinite(options.laplace_kl_residual_tol)) {
      options.laplace_kl_residual_tol = 1e-3;
    }
    options.maxiter_bfgs = std::max<std::size_t>(1, options.maxiter_bfgs);
    const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
    options.K = force_odd_K_(std::min(options.K, int_max));
    options.K_max = force_odd_K_(std::min(options.K_max, int_max));
    // The transport phase consumes warmup draws without adapting, so a
    // transport longer than warmup silently disables every adaptation.
    options.initial_transport_steps =
      std::min(options.initial_transport_steps, options.warmup);
    options.K_windowsize = std::max<std::size_t>(1, options.K_windowsize);
    options.K_windowscale = std::max<std::size_t>(1, options.K_windowscale);
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    if (!std::isfinite(options.l)) {
      options.l = 0.0;
    }
    options.J = std::clamp(options.J, Eigen::Index{0}, D);
    options.direction_lowrank_weight =
      std::isfinite(options.direction_lowrank_weight) ?
      std::clamp(options.direction_lowrank_weight, 0.0, 1.0) : 1.0;
    options.direction_min_diag_fraction =
      std::isfinite(options.direction_min_diag_fraction) ?
      std::clamp(options.direction_min_diag_fraction, 0.0, 1.0) : 0.1;
    options.pca_freeze_fraction =
      std::isfinite(options.pca_freeze_fraction) ?
      std::clamp(options.pca_freeze_fraction, 0.0, 1.0) : 0.1;
    options.transport_cov_shrink =
      std::isfinite(options.transport_cov_shrink) ?
      std::clamp(options.transport_cov_shrink, 0.0, 1.0) : 0.25;
    options.transport_cov_ratio_cap =
      std::isfinite(options.transport_cov_ratio_cap) ?
      std::max(options.transport_cov_ratio_cap, 1.0) : 4.0;
    if (std::isfinite(options.transport_max_endpoint_from_best_drop)) {
      options.transport_max_endpoint_from_best_drop =
        std::max(0.0, options.transport_max_endpoint_from_best_drop);
    } else {
      options.transport_max_endpoint_from_best_drop =
        positive_infinity;
    }
    return options;
  }

  static KAdaptationConfig K_adaptation_config_(
      const KlhrOptions& options,
      const Eigen::Index dim) {
    return {
      .enabled = options.adapt_K,
      .initial_K = options.K,
      .maximum_K = options.K_max,
      .warmup_steps = post_transport_warmup_steps(
        options.warmup, options.initial_transport_steps),
      .windowsize = options.K_windowsize,
      .windowscale = options.K_windowscale,
      .dimension = static_cast<std::size_t>(std::max<Eigen::Index>(1, dim)),
      .tolerance = options.tol,
    };
  }

  static ReflectedTransportOptions transport_options_(
      const KlhrOptions& options) {
    return {
      .N = options.N,
      .J = options.J,
      .tol = options.tol,
      .grad_clip = options.grad_clip,
      .gtol = options.gtol,
      .maxiter_bfgs = options.transport_maxiter_bfgs,
      .pca_l = options.l,
      .covariance_shrink = options.transport_cov_shrink,
      .covariance_ratio_cap = options.transport_cov_ratio_cap,
      .max_reflections = options.transport_max_reflections,
      .initial_distance = options.transport_initial_distance,
      .min_distance = options.transport_min_distance,
      .max_distance = options.transport_max_distance,
      .max_logp_drop = options.transport_max_logp_drop,
      .max_segment_logp_drop = options.transport_max_segment_logp_drop,
      .max_endpoint_from_best_drop =
        options.transport_max_endpoint_from_best_drop,
      .direction_persistence = options.transport_direction_persistence,
      .failure_direction_decay = options.transport_failure_direction_decay,
      .reflection_budget =
        options.transport_reflection_budget_per_step == 0 ? 0 :
        options.transport_reflection_budget_per_step *
        options.initial_transport_steps,
    };
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  ReflectedTransport transport_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  KWindowedAdaptation K_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  mcmcpp::WelfordAccumulator K_online_moments_;
  OnlinePCA online_pca_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::VectorXd K_center_;
  Eigen::VectorXd K_variance_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
  Eigen::MatrixXd projection_basis_;
  Eigen::MatrixXd mean_direction_basis_;
  Eigen::VectorXd mean_direction_weights_;
  bool projection_basis_ready_ = false;
  bool mean_direction_ready_ = false;
  bool mean_direction_whitened_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  bool pca_calibration_enabled_ = false;
  bool K_radius_ready_ = false;
  std::size_t pca_freeze_draw_ = 0;
  std::size_t projected_pair_count_ = 0;

  std::size_t draw_;
  // Counts only Metropolis steps, so acceptance_rate_ is a running mean over
  // the draws that actually had an accept/reject decision.
  std::size_t kl_steps_ = 0;

  struct KlStepDiagnostics {
    double xi = 0.0;
    double acceptance_probability = 0.0;
    bool valid = false;
    bool accepted = false;
  };

  // Ordered overrelaxation is reversible with respect to the distribution it
  // is applied to (Neal 1995, section 4.2). Fitting one density per *line*
  // rather than one per endpoint therefore cancels the overrelaxation kernel
  // out of the Hastings ratio entirely, leaving the importance ratio
  // pi(t1)q(t0) / pi(t0)q(t1). That needs the fit to be a function of the
  // line alone, so the fit is centred on the projection of the adaptation
  // mean, which is the same point whichever end of the line we stand at.
  KlStepDiagnostics kl_step_(const Eigen::VectorXd& rho) {
    KlStepDiagnostics diagnostics;
    auto update_acceptance = [this](const bool accepted) {
      ++kl_steps_;
      const double d = accepted - acceptance_rate_;
      acceptance_rate_ += d / static_cast<double>(kl_steps_);
    };

    const double t0 = line_coordinate_(theta_, rho);
    if (!std::isfinite(t0)) {
      update_acceptance(false);
      return diagnostics;
    }
    const Eigen::VectorXd center = theta_ - t0 * rho;

    const Eigen::VectorXd eta = fit_line_(center, rho);
    const double t1 = overrelaxed_proposal_(eta, t0);
    const double xi = t1 - t0;
    diagnostics.xi = xi;
    if (!std::isfinite(xi)) {
      update_acceptance(false);
      return diagnostics;
    }

    const Eigen::VectorXd thetap = xi * rho + theta_;
    if (!thetap.allFinite()) {
      update_acceptance(false);
      return diagnostics;
    }

    const double ldp = bsm_.log_density_noe(thetap);
    ++nfev_;
    if (!std::isfinite(ldp)) {
      update_acceptance(false);
      return diagnostics;
    }

    const double log_q0 = log_line_density_(t0, eta);
    const double log_q1 = log_line_density_(t1, eta);
    if (!std::isfinite(log_q0) || !std::isfinite(log_q1)) {
      update_acceptance(false);
      return diagnostics;
    }

    const double a = ldp - log_density_ + log_q0 - log_q1;
    if (!std::isfinite(a)) {
      update_acceptance(false);
      return diagnostics;
    }
    diagnostics.valid = true;
    diagnostics.acceptance_probability =
      a >= 0.0 ? 1.0 : std::exp(a);
    const double log_u = std::log(std_uniform_(rng_));
    diagnostics.accepted = a >= 0.0 || log_u < a;
    update_acceptance(diagnostics.accepted);
    if (diagnostics.accepted) {
      theta_ = thetap;
      log_density_ = ldp;
    }
    return diagnostics;
  }

  // Position of theta along the line, measured from the projection of the
  // adaptation mean. Depends only on the line and on frozen adaptation
  // state, never on which endpoint is current.
  double line_coordinate_(const Eigen::VectorXd& theta,
                          const Eigen::VectorXd& rho) const {
    if (mean_.size() != theta.size()) {
      return theta.dot(rho);
    }
    return (theta - mean_).dot(rho);
  }

  Eigen::VectorXd sanitize_K_variance_(Eigen::VectorXd variance) const {
    std::vector<double> positive;
    positive.reserve(static_cast<std::size_t>(variance.size()));
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (std::isfinite(variance(d)) && variance(d) > 0.0) {
        positive.push_back(variance(d));
      }
    }
    double typical = 1.0;
    if (!positive.empty()) {
      const auto middle = positive.begin() + positive.size() / 2;
      std::nth_element(positive.begin(), middle, positive.end());
      typical = *middle;
    }
    const double floor = std::max(opts_.tol, 1e-6 * typical);
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (!std::isfinite(variance(d)) || variance(d) <= 0.0) {
        variance(d) = std::max(typical, floor);
      } else {
        variance(d) = std::max(variance(d), floor);
      }
    }
    return variance;
  }

  void initialize_K_window_metric_(const bool radius_ready) {
    K_center_ = theta_;
    K_variance_ = sanitize_K_variance_(diagonal_variance_());
    K_online_moments_.reset();
    K_radius_ready_ = radius_ready;
  }

  void update_K_window_metric_() {
    if (K_online_moments_.count() > 2) {
      Eigen::VectorXd center = K_online_moments_.mean();
      Eigen::VectorXd variance = K_online_moments_.variance();
      if (center.allFinite() && variance.allFinite()) {
        K_center_ = std::move(center);
        K_variance_ = sanitize_K_variance_(std::move(variance));
        K_radius_ready_ = true;
      }
    }
    K_online_moments_.reset();
  }

  double K_standardized_squared_jump_(
      const Eigen::VectorXd& rho,
      const double xi) const {
    if (!std::isfinite(xi) || K_variance_.size() != rho.size()) {
      return 0.0;
    }
    const double direction_precision =
      (rho.array().square() / K_variance_.array()).sum();
    const double jump = xi * xi * direction_precision /
      static_cast<double>(std::max<Eigen::Index>(1, dim()));
    return std::isfinite(jump) && jump >= 0.0 ? jump : 0.0;
  }

  double K_radius_() const {
    if (!K_radius_ready_ || K_center_.size() != theta_.size() ||
        K_variance_.size() != theta_.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd centered = theta_ - K_center_;
    const double radius_squared =
      (centered.array().square() / K_variance_.array()).sum();
    if (!std::isfinite(radius_squared) || radius_squared < 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double D = static_cast<double>(std::max<Eigen::Index>(1, dim()));
    return std::log1p(radius_squared / D);
  }

  void adapt_K_warmup_(const Eigen::VectorXd& rho,
                       const KlStepDiagnostics& diagnostics) {
    K_online_moments_.update(theta_);
    const double radius = K_radius_();
    const KAdaptationObservation observation = {
      .acceptance_probability = diagnostics.acceptance_probability,
      .standardized_squared_jump =
        K_standardized_squared_jump_(rho, diagnostics.xi),
      .valid = diagnostics.valid,
      .log_density = log_density_,
      .radius_available = K_radius_ready_ && std::isfinite(radius),
      .radius = radius,
    };
    const KAdaptationUpdate update = K_adaptation_.observe(observation);
    opts_.K = update.K;
    if (update.window_closed && !update.finalized) {
      update_K_window_metric_();
    }
  }

  void reset_adaptation_to_defaults_(const bool preserve_window_position) {
    const Eigen::Index D = dim();
    // Seed from the current draw rather than the origin: the mean anchors
    // the canonical line origin in kl_step_ and centres the PCA, and on a
    // badly scaled posterior the origin can be many standard deviations from
    // anywhere the chain will ever be.
    mean_ = (theta_.size() == D && theta_.allFinite()) ?
      theta_ : Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_weights_ = Eigen::VectorXd::Ones(opts_.J);

    online_moments_.reset();
    online_pca_.reset();
    projected_moments_.reset();
    if (!preserve_window_position) {
      windowed_adaptation_.reset();
    }

    projection_basis_ready_ = false;
    mean_direction_ready_ = false;
    mean_direction_whitened_ = false;
    pca_frozen_ = false;
    lowrank_ready_ = false;
    projected_pair_count_ = 0;
  }

  Eigen::VectorXd diagonal_variance_() const {
    Eigen::VectorXd variance = cov_;
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (!std::isfinite(variance(d)) || variance(d) <= 0.0) {
        variance(d) = 1.0;
      } else {
        variance(d) = std::max(variance(d), opts_.tol);
      }
    }
    return variance;
  }

  Eigen::VectorXd metric_scale_() const {
    return diagonal_variance_().array().sqrt().matrix();
  }

  Eigen::VectorXd direction_noise_() {
    const Eigen::Index D = dim();
    const double alpha = opts_.direction_lowrank_weight;
    const double min_diag_fraction = opts_.direction_min_diag_fraction;
    const Eigen::VectorXd base_var = diagonal_variance_();
    Eigen::VectorXd residual_var = base_var;
    const Eigen::Index rank = lowrank_ready_ ? opts_.J : 0;
    if (rank == 0 || alpha == 0.0) {
      return diagonal_direction_noise_(base_var);
    }
    const Eigen::MatrixXd lowrank_basis = eigvecs_.leftCols(rank);
    const Eigen::VectorXd lowrank_variance = eigvals_.head(rank);
    residual_var -= alpha *
      (lowrank_basis.array().square().matrix() * lowrank_variance);

    for (Eigen::Index d = 0; d < D; ++d) {
      const double floor = std::max(opts_.tol, min_diag_fraction * base_var(d));
      if (!std::isfinite(residual_var(d)) || residual_var(d) < floor) {
        residual_var(d) = floor;
      }
    }

    Eigen::VectorXd noise = diagonal_direction_noise_(residual_var);
    const Eigen::VectorXd lowrank_sd =
      (alpha * lowrank_variance.array()).sqrt().matrix();
    noise += lowrank_basis *
      lowrank_sd.cwiseProduct(normal_rng_(rank));

    if (!noise.allFinite()) {
      return normal_rng_(D);
    }
    return noise;
  }

  Eigen::VectorXd diagonal_direction_noise_() {
    return diagonal_direction_noise_(diagonal_variance_());
  }

  Eigen::VectorXd diagonal_direction_noise_(
      const Eigen::VectorXd& variance) {
    return variance.array().sqrt().matrix().cwiseProduct(
      normal_rng_(dim()));
  }

  Eigen::VectorXd mean_direction_noise_() {
    const Eigen::Index D = dim();
    if (!mean_direction_ready_) {
      return diagonal_direction_noise_();
    }

    std::discrete_distribution<Eigen::Index> component(
      mean_direction_weights_.data(),
      mean_direction_weights_.data() + mean_direction_weights_.size());
    const Eigen::Index j = component(rng_);

    Eigen::VectorXd noise;
    if (mean_direction_whitened_) {
      noise = normal_rng_(D);
      noise += mean_direction_basis_.col(j);
      noise = metric_scale_().array() * noise.array();
    } else {
      noise = diagonal_direction_noise_();
      noise += mean_direction_basis_.col(j);
    }
    return noise;
  }

  void initialize_pca_schedule_() {
    std::size_t final_start = 0;
    const auto& closures = windowed_adaptation_.closures();
    if (opts_.warmup > 0) {
      final_start = 1;
      if (closures.size() >= 2 && closures.back() == opts_.warmup) {
        final_start = closures[closures.size() - 2] + 1;
      }
    }

    const std::size_t final_length = final_start <= opts_.warmup ?
      opts_.warmup - final_start + 1 : 0;
    // Zero-valued PCA options intentionally disable low-rank calibration.
    pca_calibration_enabled_ =
      opts_.J > 0 && opts_.pca_freeze_fraction > 0.0 && final_length > 2;
    if (!pca_calibration_enabled_) {
      pca_freeze_draw_ = opts_.warmup;
      return;
    }

    const auto tail = static_cast<std::size_t>(
      std::ceil(opts_.pca_freeze_fraction * final_length));
    const std::size_t tail_length =
      std::clamp<std::size_t>(tail, 1, final_length);
    pca_freeze_draw_ =
      std::max(final_start, opts_.warmup - tail_length);
  }

  bool set_projection_basis_from_online_pca_() {
    if (online_pca_.count() < opts_.J) {
      return false;
    }

    Eigen::MatrixXd basis = online_pca_.vectors();
    if (!basis.allFinite()) {
      return false;
    }

    projection_basis_ = basis.leftCols(opts_.J);
    for (Eigen::Index j = 0; j < projection_basis_.cols(); ++j) {
      const double norm = projection_basis_.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        projection_basis_ready_ = false;
        return false;
      }
      projection_basis_.col(j) /= norm;
    }
    projection_basis_ready_ = true;
    return true;
  }

  bool set_mean_direction_(Eigen::MatrixXd basis,
                           Eigen::VectorXd weights,
                           const bool whitened) {
    mean_direction_ready_ = false;
    mean_direction_whitened_ = false;
    if (!basis.allFinite() || !weights.allFinite()) {
      mean_direction_basis_.setZero();
      mean_direction_weights_.setOnes();
      return false;
    }

    for (Eigen::Index j = 0; j < basis.cols(); ++j) {
      const double norm = basis.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        mean_direction_basis_.setZero();
        mean_direction_weights_.setOnes();
        return false;
      }
      basis.col(j) /= norm;
    }

    mean_direction_basis_ = std::move(basis);
    mean_direction_weights_ = weights.cwiseMax(opts_.tol);
    mean_direction_ready_ = true;
    mean_direction_whitened_ = whitened;
    return true;
  }

  void set_mean_direction_from_online_pca_(const bool whitened = false) {
    if (online_pca_.count() < opts_.J) {
      return;
    }

    Eigen::MatrixXd basis = online_pca_.vectors().leftCols(opts_.J);
    Eigen::VectorXd weights = online_pca_.values().head(opts_.J);
    (void) set_mean_direction_(std::move(basis), std::move(weights), whitened);
  }

  void update_projected_moments_(const Eigen::VectorXd& theta) {
    if (!projection_basis_ready_) {
      return;
    }

    const Eigen::VectorXd projected = projection_basis_.transpose() * theta;
    if (projected.allFinite()) {
      projected_moments_.update(projected);
    }
  }

  void activate_projected_pair_() {
    if (!projection_basis_ready_ || projected_moments_.count() <= 2) {
      return;
    }

    Eigen::VectorXd variances = projected_moments_.variance();
    for (Eigen::Index j = 0; j < opts_.J; ++j) {
      if (!std::isfinite(variances(j)) || variances(j) <= opts_.tol) {
        variances(j) = opts_.tol;
      }
    }

    eigvecs_.leftCols(opts_.J) = projection_basis_;
    eigvals_.head(opts_.J) = variances;
    ++projected_pair_count_;
    lowrank_ready_ = projected_pair_count_ >= 2;
  }

  void freeze_pca_for_final_calibration_() {
    if (pca_frozen_) {
      return;
    }

    // Bank the variances accumulated against the outgoing basis before
    // installing a new one, otherwise a whole window's worth of projected
    // moments is discarded in favour of the short calibration tail.
    activate_projected_pair_();

    const bool frozen = set_projection_basis_from_online_pca_();
    set_mean_direction_from_online_pca_();
    if (!frozen && projected_pair_count_ > 0) {
      projection_basis_ = eigvecs_.leftCols(opts_.J);
      projection_basis_ready_ = projection_basis_.allFinite();
    }

    projected_moments_.reset();
    online_pca_.reset();
    pca_frozen_ = true;
  }

  // Seeding the mean matters twice over: it centres the first post-transport
  // PCA window, and it anchors the canonical line origin used by kl_step_.
  // It has to happen on the rollback path too, where the adaptation state is
  // otherwise reset to its defaults.
  void seed_mean_from_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    if (handoff.mean.size() == dim() && handoff.mean.allFinite()) {
      mean_ = handoff.mean;
    } else if (handoff.state.theta.size() == dim() &&
               handoff.state.theta.allFinite()) {
      mean_ = handoff.state.theta;
    }
  }

  void apply_transport_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    theta_ = handoff.state.theta;
    log_density_ = handoff.state.log_density;
    if (handoff.rollback) {
      reset_adaptation_to_defaults_(true);
      seed_mean_from_handoff_(handoff);
      initialize_K_window_metric_(false);
      return;
    }

    if (handoff.covariance.allFinite()) {
      cov_ = handoff.covariance;
    }
    seed_mean_from_handoff_(handoff);
    if (!handoff.pca_ready ||
        !set_mean_direction_(handoff.pca_basis, handoff.pca_weights,
                             handoff.pca_whitened)) {
      mean_direction_basis_.setZero();
      mean_direction_weights_.setOnes();
      mean_direction_ready_ = false;
      mean_direction_whitened_ = false;
    }
    online_pca_.reset();
    initialize_K_window_metric_(false);
  }

  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    if (adaptation_draw > opts_.warmup) {
      return;
    }

    if (pca_calibration_enabled_ && !pca_frozen_ &&
        adaptation_draw >= pca_freeze_draw_) {
      freeze_pca_for_final_calibration_();
    }

    online_moments_.update(theta);
    if (pca_calibration_enabled_) {
      update_projected_moments_(theta);
      if (!pca_frozen_) {
        online_pca_.update(theta - mean_);
      }
    }

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();

      if (pca_calibration_enabled_ && !pca_frozen_) {
        activate_projected_pair_();
        const bool has_next_basis = set_projection_basis_from_online_pca_();
        set_mean_direction_from_online_pca_();
        projected_moments_.reset();
        if (!has_next_basis) {
          projection_basis_ready_ = false;
        }
        online_pca_.reset();
      }
    }

    if (pca_calibration_enabled_ && pca_frozen_ &&
        adaptation_draw == opts_.warmup) {
      activate_projected_pair_();
    }
  }

  void set_bad_kl_(const Eigen::VectorXd& eta, double& value,
                   Eigen::VectorXd& grad) const {
    numerics::set_bad_kl(eta, value, grad);
  }

  double relative_log_scale_(const double raw, const double log_s0) const {
    return numerics::relative_log_scale(raw, log_s0);
  }

  double relative_log_scale_derivative_(const double raw) const {
    return numerics::relative_log_scale_derivative(raw);
  }

  double scale_from_log_(double log_s) const {
    return numerics::scale_from_log(log_s, opts_.tol);
  }

  double overrelaxed_proposal_impl_(const double u_raw) {
    // A saturated CDF value would send the overrelaxation kernel a clamped
    // surrogate, breaking its reversibility with respect to the fitted
    // density. Fall back to an independent draw, which is exactly
    // reversible for any K.
    if (opts_.K == 0 || numerics::probability_saturated(u_raw)) {
      return clamp_probability_(std_uniform_(rng_));
    }
    const double u = clamp_probability_(u_raw);

    const int K = opts_.K;
    std::binomial_distribution<int> binomial(K, u);
    const int r = binomial(rng_);

    double up = u;
    if (r > K - r) {
      const double v = beta_rng_(K - r + 1.0, 2.0 * r - K);
      up = u * v;
    } else if (r < K - r) {
      const double v = beta_rng_(r + 1.0, K - 2.0 * r);
      up = 1.0 - (1.0 - u) * v;
    }

    return clamp_probability_(up);
  }

  double beta_rng_(double a, double b) {
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);

    double x = gamma_a(rng_);
    double y = gamma_b(rng_);
    double total = x + y;
    while (!std::isfinite(total) || total <= 0.0) {
      x = gamma_a(rng_);
      y = gamma_b(rng_);
      total = x + y;
    }
    return x / total;
  }

  static double clamp_probability_(double p) {
    return numerics::clamp_probability(p);
  }

  static double normal_cdf_(double z) {
    return normal_cdf(z);
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

};

} // namespace klhr
