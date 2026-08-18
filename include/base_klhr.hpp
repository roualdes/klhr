#pragma once

#include "bfgs.hpp"
#include "gausshermite.hpp"
#include "k_adaptation.hpp"
#include "klhr_numerics.hpp"
#include "normal_quantile.hpp"
#include "onlinepca.hpp"
#include "reflected_transport.hpp"
#include "sketch.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>
#include <welford.hpp>
#include <windowedadaptation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
  std::size_t K = 15;  // Odd K only
  bool adapt_K = true;
  std::size_t K_max = 511;
  std::size_t K_windowsize = 50;
  std::size_t K_windowscale = 2;
  std::size_t K_refresh_lags = 4;
  bool K_interleave_trials = true;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  Eigen::Index J = 10;
  double direction_lowrank_weight = 1.0;
  double direction_min_diag_fraction = 0.1;
  bool lowrank_during_warmup = true;
  double pca_freeze_fraction = 0.1;
  // Window closures required before the OnlinePCA low-rank direction is used.
  // Inert under the default mixture_direction = true, which routes adaptation
  // to the sketch instead: projected_moments_ never accumulates, so
  // activate_projected_pair_ returns early every time and the count never
  // leaves zero. It is still a live --lowrank-min-activations flag in
  // examples/example.cpp, so hardcoding it means removing that flag too -- and
  // the honest version of that change removes the whole OnlinePCA path, which
  // the default configuration no longer reaches.
  std::size_t lowrank_min_activations = 1;
  Eigen::Index sketch_columns = 60;
  Eigen::Index transport_J = 1;
  // Mix a diagonal direction with two spectral components. POSITION learns
  // broad directions from draws. CURVATURE diagonalises BridgeStan's local
  // negative log-density Hessian, so it can find a broad mode before the
  // chain traverses it. Both feed the same trace-controlled direction law
  // after learning their respective spectral pairs.
  bool mixture_direction = true;
  bool position_sketch_direction = true;
  bool curvature_sketch_direction = true;
  // Position and curvature have separate caps because the number of broad
  // local-curvature modes need not match the estimable covariance rank.
  Eigen::Index sketch_max_rank = 10;
  Eigen::Index curvature_max_rank = 10;
  // Retain only standardized position-covariance or inverse-curvature
  // eigenvalues whose scale is at least this far above the diagonal baseline.
  double sketch_eigenvalue_cutoff = 2.0;
  // Fraction of a sketch law's total standardized variance left in its
  // full-rank isotropic residual. The remaining trace is assigned to the
  // retained subspace, independent of ambient dimension.
  double sketch_residual_fraction = 0.02;
  // Mixture floors. The diagonal floor is structural, and deliberately not
  // small: on a target with no correlation to find (ill-normal) the diagonal
  // is the correct answer, standardized ESJD is exactly flat so the bandit
  // gets no gradient, and the floor is what keeps a useless component from
  // holding weight indefinitely.
  double mixture_floor_diagonal = 0.25;
  double mixture_floor_position = 0.02;
  double mixture_floor_curvature = 0.02;
  // Log-odds prior the weights relax toward when rewards are uninformative.
  double mixture_prior_position = -1.0;
  double mixture_prior_curvature = -1.0;
  double bandit_gamma0 = 0.5;
  double bandit_decay = 0.6;
  // Pull toward the prior. At 0.1 the cumulative pull over a run was about
  // 0.2 in log-odds against a prior of -1, so an uninformative component kept
  // most of its starting mass: on ill-normal the diagonal fired only 45% of
  // the time and the mixture came out 42% worse than diagonal-only. The
  // reward term is O(1) per update, so a genuinely useful component still
  // overcomes this easily.
  double bandit_shrink = 0.5;
  std::size_t bandit_min_draws = 20;
  // Terminal buffer: the final warmup draws, over which every slow and fast
  // parameter is frozen so the chain equilibrates to the kernel that will
  // actually sample. Without it the last metric change lands on the final
  // warmup draw.
  std::size_t terminal_buffer = 50;
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
  // Reflections the transport phase may spend, quoted per step but pooled
  // across the phase. transport_options_ multiplies it by
  // initial_transport_steps into a single ReflectedTransportOptions
  // ::reflection_budget, and each step then takes
  // max(1, min(transport_max_reflections, budget - used)). So a step is free
  // to reflect far past 75 as long as other steps spend less, and only the
  // phase total is bounded. That pooling is the point: the long excursions
  // that carry a badly initialised chain to the bulk are rare, and a strict
  // per-step cap would truncate exactly those. Zero drops the pooled bound and
  // leaves only transport_max_reflections per step.
  std::size_t transport_reflection_budget_per_step = 75;
};

class BaseKLHR {
public:
  static constexpr int kDiag = 0;
  static constexpr int kPosition = 1;
  static constexpr int kCurvature = 2;
  static constexpr int kComponents = 3;
  // The direction that was actually used came from no component's law, so no
  // component may be credited or charged for how it performed.
  static constexpr int kUnattributed = -1;

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
    adaptation_steps_(post_transport_warmup_steps(
      opts_.warmup, opts_.initial_transport_steps)),
    adaptation_span_(adaptation_span_of_(adaptation_steps_,
                                         opts_.terminal_buffer)),
    transport_(bsm_.dim(), transport_options_(opts_)),
    windowed_adaptation_(adaptation_span_, opts_.windowsize,
                         opts_.windowscale),
    K_adaptation_(K_adaptation_config_(opts_, bsm_.dim())),
    online_moments_(bsm_.dim()),
    K_online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), opts_.J, opts_.l, opts_.tol),
    position_sketch_(bsm_.dim(),
                     opts_.mixture_direction &&
                       opts_.position_sketch_direction ?
                         opts_.sketch_columns : 0,
                     opts_.tol, opts_.seed ^ 0x5bf03635U),
    curvature_direction_(bsm_.dim(), opts_.tol),
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

  // The two names are not redundant: diagonal_variance_() says what the value
  // is (the sanitised diagonal of cov_), metric_variance() says what it is
  // for. Renaming the private one would lose the distinction between the
  // diagonal and the full metric, which the low-rank components also feed.
  Eigen::VectorXd metric_variance() const { return diagonal_variance_(); }
  Eigen::VectorXd transport_covariance() const { return transport_cov_; }

  // Diagnostics are paired in component order: POSITION, CURVATURE.
  std::array<Eigen::Index, 2> sketch_ranks() const {
    return {position_sketch_.rank(), curvature_direction_.rank()};
  }
  std::array<double, 2> sketch_mean_ranks() const {
    return {position_sketch_.mean_rank(), curvature_direction_.mean_rank()};
  }
  std::array<std::size_t, 2> sketch_dropouts() const {
    return {position_sketch_.dropouts(), curvature_direction_.dropouts()};
  }

  // Final mixture weights, in component order: DIAG, POSITION, CURVATURE.
  //
  // To judge whether the mixture is worth its cost, read a weight against the
  // floor it cannot go below (mixture_floor_*) and against the prior it decays
  // to when rewards are uninformative (mixture_prior_*). A component parked at
  // its floor was tried and rejected by the bandit; a component at zero did
  // not find a spectral mode beyond the configured cutoff.
  //
  // A weight well above its prior is the bandit reporting a real per-coordinate
  // ESJD gain, but that is its own objective, not the run's -- confirm it
  // against standardized RMSE or min-ESS on the model in question before
  // reading a large weight as a win.
  std::array<double, kComponents> mixture_weights() const {
    return mixture_p_;
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
      if (draw_ == opts_.initial_transport_steps) {
        apply_transport_handoff_(transport_.finish(rng_, std_normal_));
      }
      return bsm_.param_constrain(theta_);
    }

    const std::size_t K_draw = post_transport_warmup_draw(
      draw_, opts_.warmup, opts_.initial_transport_steps);
    const bool adapting = adapting_(K_draw);
    if (adapting && K_adaptation_.enabled()) {
      opts_.K = K_adaptation_.K();
    }
    Eigen::VectorXd rho = random_direction();
    const KlStepDiagnostics diagnostics = kl_step_(rho);
    if (adapting) {
      bandit_observe_(rho, diagnostics);
      if (K_adaptation_.enabled()) {
        adapt_K_warmup_(rho, diagnostics);
      }
    }
    adapt_warmup_(theta_, K_draw);
    return bsm_.param_constrain(theta_);
  }

  Eigen::VectorXd random_direction() {
    const Eigen::Index D = dim();
    // One direction model throughout: a Gaussian shaped like the current
    // covariance estimate, degrading to its diagonal when no low-rank basis
    // is available yet.
    const bool lowrank_allowed =
      opts_.lowrank_during_warmup || draw_ > opts_.warmup;
    Eigen::VectorXd rho;
    if (opts_.mixture_direction) {
      rho = mixture_direction_noise_();
    } else if (lowrank_allowed && lowrank_ready_) {
      rho = direction_noise_();
    } else {
      rho = diagonal_direction_noise_();
    }

    // Both salvage paths below replace the chosen component's draw with
    // something that is not its law -- isotropic noise, or a coordinate axis.
    // Crediting the resulting jump to the component that was chosen would feed
    // the bandit a reward it did not earn, and crediting it to DIAG is no
    // better, since neither is the diagonal law either. Leave it unattributed
    // and let bandit_observe_ drop the draw.
    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      last_component_ = kUnattributed;
      rho = normal_rng_(D);
      norm = rho.norm();
    }

    if (std::isfinite(norm) && norm > 0.0) {
      rho /= norm;
    } else {
      last_component_ = kUnattributed;
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

  virtual double log_line_density_(const double t,
                                   const Eigen::VectorXd& eta) const = 0;

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
    bool curvature_usable = false;
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
    double inverse_curvature = std::numeric_limits<double>::quiet_NaN();
    if (mode.hess_inv.rows() == 1 && mode.hess_inv.cols() == 1) {
      inverse_curvature = mode.hess_inv(0, 0);
    }
    out.curvature_usable =
      std::isfinite(inverse_curvature) && inverse_curvature > 0.0;
    if (out.curvature_usable) {
      out.log_scale = 0.5 * std::log(inverse_curvature);
    }
    out.success = mode.success;
    return out;
  }

  bfgs::BfgsOptions bfgs_options_() const {
    return {.gtol = opts_.gtol,
            .xrtol = opts_.gtol,
            .maxiter_bfgs = opts_.maxiter_bfgs};
  }


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

    if (!mode.success || !mode.curvature_usable) {
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

  // K=1 is the independence kernel; identical to K=0. Even K adds an
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
    // Fallbacks match the declared defaults; a rejected value must land on the
    // documented setting, not on a different one.
    if (!(options.gtol > 0.0) || !std::isfinite(options.gtol)) {
      options.gtol = KlhrOptions{}.gtol;
    }
    if (!(options.laplace_kl_residual_tol >= 0.0) ||
        !std::isfinite(options.laplace_kl_residual_tol)) {
      options.laplace_kl_residual_tol = KlhrOptions{}.laplace_kl_residual_tol;
    }
    options.maxiter_bfgs = std::max<std::size_t>(1, options.maxiter_bfgs);
    options.lowrank_min_activations =
      std::max<std::size_t>(1, options.lowrank_min_activations);
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
    options.transport_J = std::clamp(options.transport_J, Eigen::Index{0}, D);
    // Probes beyond D buy nothing and cost more than the matrix they avoid.
    //
    // The Nystrom identity S ~ Y (Omega' S Omega)^+ Y' is exact once Omega
    // spans R^D, so for m >= D there is no approximation error left to reduce
    // and the extra columns are pure overhead.
    options.sketch_columns = std::clamp(options.sketch_columns,
                                        Eigen::Index{1},
                                        std::max<Eigen::Index>(D, 1));
    options.sketch_max_rank = std::clamp(options.sketch_max_rank,
                                         Eigen::Index{1},
                                         std::max<Eigen::Index>(D, 1));
    options.curvature_max_rank = std::clamp(options.curvature_max_rank,
                                            Eigen::Index{1},
                                            std::max<Eigen::Index>(D, 1));
    options.sketch_eigenvalue_cutoff =
      std::isfinite(options.sketch_eigenvalue_cutoff) &&
      options.sketch_eigenvalue_cutoff > 1.0 ?
        options.sketch_eigenvalue_cutoff : 2.0;
    options.sketch_residual_fraction =
      std::isfinite(options.sketch_residual_fraction) ?
        std::clamp(options.sketch_residual_fraction,
                   std::max(options.tol, 1e-12), 1.0) :
        KlhrOptions{}.sketch_residual_fraction;
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
      .warmup_steps = adaptation_span_of_(
        post_transport_warmup_steps(
          options.warmup, options.initial_transport_steps),
        options.terminal_buffer),
      .windowsize = options.K_windowsize,
      .windowscale = options.K_windowscale,
      .dimension = static_cast<std::size_t>(std::max<Eigen::Index>(1, dim)),
      .refresh_lags = options.K_refresh_lags,
      .interleave_trials = options.K_interleave_trials,
      .tolerance = options.tol,
    };
  }

  static ReflectedTransportOptions transport_options_(
      const KlhrOptions& options) {
    return {
      .N = options.N,
      .J = options.transport_J,
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
  // Warmup draws that actually reach the adaptation, i.e. after transport.
  std::size_t adaptation_steps_;
  // Post-transport draws that actually adapt, i.e. excluding the terminal
  // buffer. Windows and the K ladder are laid out over this, so everything is
  // already frozen before the buffer begins.
  std::size_t adaptation_span_;
  ReflectedTransport transport_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  KWindowedAdaptation K_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  mcmcpp::WelfordAccumulator K_online_moments_;
  OnlinePCA online_pca_;
  PositionSketch position_sketch_;
  CurvatureDirection curvature_direction_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  // sanitize_variance_(cov_), kept in step by set_covariance_.
  Eigen::VectorXd diagonal_variance_cache_;
  Eigen::VectorXd K_center_;
  Eigen::VectorXd K_variance_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
  Eigen::MatrixXd projection_basis_;
  Eigen::VectorXd transport_cov_;
  // Mixture direction law and its bandit.
  // Initialised to the prior by reset_mixture_(), which the constructor runs.
  std::array<double, kComponents> mixture_logw_{{0.0, 0.0, 0.0}};
  std::array<double, kComponents> mixture_p_{{1.0, 0.0, 0.0}};
  struct BanditBlock {
    std::array<Eigen::VectorXd, kComponents> sum;
    std::array<std::size_t, kComponents> cnt{{0, 0, 0}};
  };
  std::map<std::size_t, BanditBlock> blocks_;
  std::size_t bandit_updates_ = 0;
  int last_component_ = 0;

  bool projection_basis_ready_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  bool pca_calibration_enabled_ = false;
  bool K_radius_ready_ = false;
  std::size_t pca_freeze_draw_ = 0;
  std::size_t projected_pair_count_ = 0;

  std::size_t draw_;
  std::size_t kl_steps_ = 0;

  // Still needed: draw() hands this to bandit_observe_ and adapt_K_warmup_,
  // which between them read xi, acceptance_probability and valid. `accepted`
  // is the exception -- kl_step_ is its only reader, so it could be a local.
  // Left on the struct because a rejected step and an invalid one are
  // different things and having both fields visible keeps that legible at the
  // call sites.
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

  // Floor relative to a typical variance, not absolutely. A coordinate that
  // happens not to move inside a window can report a variance many orders
  // below the rest; an absolute floor of tol lets it through, and everything
  // that divides by it -- the direction draw, the metric scale, the bandit
  // reward -- is then dominated by that one coordinate.
  Eigen::VectorXd sanitize_variance_(Eigen::VectorXd variance) const {
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
    // Already sanitised by set_covariance_.
    K_variance_ = diagonal_variance_();
    K_online_moments_.reset();
    K_radius_ready_ = radius_ready;
  }

  void update_K_window_metric_() {
    if (K_online_moments_.count() > 2) {
      Eigen::VectorXd center = K_online_moments_.mean();
      Eigen::VectorXd variance = K_online_moments_.variance();
      if (center.allFinite() && variance.allFinite()) {
        K_center_ = std::move(center);
        K_variance_ = sanitize_variance_(std::move(variance));
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
    set_covariance_(Eigen::VectorXd::Ones(D));
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);

    online_moments_.reset();
    reset_basis_estimator_();
    projected_moments_.reset();
    if (!preserve_window_position) {
      windowed_adaptation_.reset();
    }

    projection_basis_ready_ = false;
    pca_frozen_ = false;
    lowrank_ready_ = false;
    projected_pair_count_ = 0;
    reset_mixture_();
  }

  // cov_ moves only at a window close, at the transport handoff, and on a
  // reset, so the sanitised form is cached at those three points rather than
  // rebuilt on every read. The read side is hot: the direction draw and the
  // bandit reward each want it once per draw, and sanitize_variance_ costs an
  // allocation plus an nth_element over D.
  //
  // Route every write through here so the two cannot drift apart.
  void set_covariance_(Eigen::VectorXd covariance) {
    cov_ = std::move(covariance);
    diagonal_variance_cache_ = sanitize_variance_(cov_);
  }

  const Eigen::VectorXd& diagonal_variance_() const {
    return diagonal_variance_cache_;
  }

  Eigen::VectorXd metric_scale_() const {
    return diagonal_variance_().array().sqrt().matrix();
  }

  static std::size_t adaptation_span_of_(const std::size_t steps,
                                         const std::size_t terminal) {
    // The buffer must never swallow the whole adaptation.
    if (terminal == 0 || steps <= terminal) {
      return steps;
    }
    return steps - terminal;
  }

  bool adapting_(const std::size_t adaptation_draw) const {
    return adaptation_draw > 0 && adaptation_draw <= adaptation_span_;
  }

  double mixture_floor_(const int c) const {
    if (c == kPosition) return opts_.mixture_floor_position;
    if (c == kCurvature) return opts_.mixture_floor_curvature;
    return opts_.mixture_floor_diagonal;
  }

  double mixture_prior_(const int c) const {
    if (c == kPosition) return opts_.mixture_prior_position;
    if (c == kCurvature) return opts_.mixture_prior_curvature;
    return 0.0;
  }

  // A component is live only once its block has actually been built from
  // data, which gives the "do not use an unestimated basis" gate for free.
  std::vector<int> live_components_() const {
    std::vector<int> live{kDiag};
    if (position_sketch_.ready()) live.push_back(kPosition);
    if (curvature_direction_.ready()) live.push_back(kCurvature);
    return live;
  }

  void refresh_mixture_weights_() {
    const std::vector<int> live = live_components_();
    double slack = 1.0;
    for (const int c : live) {
      slack -= mixture_floor_(c);
    }
    slack = std::max(slack, 0.0);

    double max_logw = -positive_infinity;
    for (const int c : live) {
      max_logw = std::max(max_logw, mixture_logw_[c]);
    }
    double total = 0.0;
    std::array<double, kComponents> w{{0.0, 0.0, 0.0}};
    for (const int c : live) {
      w[c] = std::exp(mixture_logw_[c] - max_logw);
      total += w[c];
    }
    mixture_p_ = {{0.0, 0.0, 0.0}};
    if (!(total > 0.0) || !std::isfinite(total)) {
      mixture_p_[kDiag] = 1.0;
      return;
    }
    for (const int c : live) {
      mixture_p_[c] = mixture_floor_(c) + slack * w[c] / total;
    }
  }

  // nu is symmetric and does not depend on theta, so nu(d)/nu(-d) is one and
  // the direction never enters the Hastings ratio.
  Eigen::VectorXd mixture_direction_noise_() {
    const double u = std_uniform_(rng_);
    double cumulative = 0.0;
    int chosen = kDiag;
    for (int c = 0; c < kComponents; ++c) {
      cumulative += mixture_p_[c];
      if (u < cumulative) {
        chosen = c;
        break;
      }
    }

    if (chosen == kPosition && position_sketch_.ready()) {
      last_component_ = kPosition;
      return position_direction_noise_();
    }
    if (chosen == kCurvature && curvature_direction_.ready()) {
      last_component_ = kCurvature;
      return curvature_direction_noise_();
    }
    last_component_ = kDiag;
    return diagonal_direction_noise_();
  }

  // Rao-Blackwellised ESJD in the frozen diagonal metric: the acceptance
  // probability rather than the 0/1 outcome, which is the same expectation
  // with far less variance.
  void bandit_observe_(const Eigen::VectorXd& rho,
                       const KlStepDiagnostics& diagnostics) {
    if (!opts_.mixture_direction || !std::isfinite(diagnostics.xi) ||
        last_component_ < 0 || last_component_ >= kComponents) {
      return;
    }
    const Eigen::VectorXd& variance = diagonal_variance_();
    // per coordinate: alpha * (xi rho_d)^2 / sigma_d^2
    const Eigen::VectorXd contribution =
      (diagnostics.acceptance_probability * diagnostics.xi * diagnostics.xi) *
      (rho.array().square() / variance.array()).matrix();
    if (!contribution.allFinite() || (contribution.array() < 0.0).any()) {
      return;
    }
    // Stratify by overrelaxation level rather than restarting on every
    // change: K interleaves in short trial blocks, so resetting there left
    // every block below bandit_min_draws and the weights never moved at all.
    BanditBlock& block = blocks_[opts_.K];
    Eigen::VectorXd& target = block.sum[last_component_];
    if (target.size() != contribution.size()) {
      target = Eigen::VectorXd::Zero(contribution.size());
    }
    target += contribution;
    ++block.cnt[last_component_];
  }

  // Close a reward block and move the weights: a replicator step against the
  // mixture-average reward, plus shrinkage toward a diagonal-favouring prior
  // so an uninformative reward decays a component rather than leaving it
  // parked wherever it started. (The two halves are not in tension -- the
  // replicator term is the gradient step and the shrinkage term is the
  // regulariser on it; both appear in the single update at the end.)
  //
  // The mixture's per-coordinate ESJD is E_d(p) = sum_c p_c e_cd, and the
  // objective is Phi(p) = -sum_d 1/E_d -- a monotone transform of the
  // harmonic mean of the E_d, and a direct proxy for -sum_d tau_d, which is
  // what standardized RMSE measures. Phi is concave in p (its Hessian is
  // -2 sum_d e_cd e_bd / E_d^3, a negatively weighted Gram), so the
  // multiplicative update below is mirror ascent on a concave objective and
  // has a unique optimum on the simplex.
  //
  // Conveniently sum_c p_c g_c = sum_d E_d / E_d^2 = sum_d 1/E_d, so the
  // baseline the replicator already divides by is -Phi itself, and the
  // update keeps exactly the shape it had under the scalar reward.
  void finalize_bandit_block_() {
    if (!opts_.mixture_direction) {
      return;
    }
    const std::vector<int> live = live_components_();
    if (live.size() <= 1) {
      blocks_.clear();
      return;
    }

    std::array<double, kComponents> ratio_sum{{0.0, 0.0, 0.0}};
    std::size_t strata = 0;
    for (const auto& [level, block] : blocks_) {
      (void) level;
      bool enough = true;
      for (const int c : live) {
        if (block.cnt[c] < opts_.bandit_min_draws ||
            block.sum[c].size() == 0) {
          enough = false;
        }
      }
      if (!enough) {
        continue;
      }

      std::array<Eigen::VectorXd, kComponents> e;
      Eigen::VectorXd mixed;
      for (const int c : live) {
        e[c] = block.sum[c] / static_cast<double>(block.cnt[c]);
        if (mixed.size() == 0) {
          mixed = Eigen::VectorXd::Zero(e[c].size());
        }
        mixed += mixture_p_[c] * e[c];
      }
      if (mixed.size() == 0 || !mixed.allFinite()) {
        continue;
      }

      // The diagonal law makes the mixture full rank. Keep a numerical floor
      // for finite-sample blocks in which a coordinate happened to receive no
      // measurable jump.
      const double floor = std::max(opts_.tol, 1e-12 * mixed.maxCoeff());
      const Eigen::ArrayXd E = mixed.array().max(floor);
      const Eigen::ArrayXd inv = 1.0 / E;
      const double baseline = inv.sum();
      std::array<double, kComponents> gradient{{0.0, 0.0, 0.0}};
      for (const int c : live) {
        gradient[c] = (e[c].array() * inv.square()).sum();
      }
      if (!std::isfinite(baseline) || baseline <= 0.0) {
        continue;
      }
      bool usable = true;
      for (const int c : live) {
        if (!std::isfinite(gradient[c])) {
          usable = false;
        }
      }
      if (!usable) {
        continue;
      }
      for (const int c : live) {
        ratio_sum[c] += gradient[c] / baseline;
      }
      ++strata;
    }

    const double gamma = opts_.bandit_gamma0 /
      std::pow(1.0 + static_cast<double>(bandit_updates_), opts_.bandit_decay);
    for (const int c : live) {
      // With no usable stratum the reward term is zero and only the prior
      // pulls, so an uninformative component decays toward the diagonal
      // instead of parking wherever it started.
      const double reward_term = strata > 0
        ? ratio_sum[c] / static_cast<double>(strata) - 1.0 : 0.0;
      mixture_logw_[c] += gamma * reward_term -
        gamma * opts_.bandit_shrink *
          (mixture_logw_[c] - mixture_prior_(c));
    }
    ++bandit_updates_;

    blocks_.clear();
    refresh_mixture_weights_();
  }

  void rebuild_mixture_() {
    if (!opts_.mixture_direction) {
      return;
    }
    const Eigen::VectorXd scale = metric_scale_();
    if (opts_.position_sketch_direction) {
      position_sketch_.refresh(scale, opts_.sketch_max_rank,
                               opts_.sketch_eigenvalue_cutoff);
    }
    if (opts_.curvature_sketch_direction) {
      refresh_curvature_direction_(scale);
    }
    refresh_mixture_weights_();
  }

  void refresh_curvature_direction_(
      const Eigen::Ref<const Eigen::VectorXd>& metric_scale) {
    Eigen::VectorXd gradient(dim());
    Eigen::VectorXd flat_hessian(dim() * dim());
    double value = 0.0;
    bsm_.log_density_gradient_hessian_noe(
      theta_, value, gradient, flat_hessian);
    ++nfev_;

    Eigen::MatrixXd negative_hessian = Eigen::MatrixXd::Constant(
      dim(), dim(), std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(value) && gradient.allFinite() &&
        flat_hessian.allFinite()) {
      const Eigen::Map<const Eigen::MatrixXd> log_density_hessian(
        flat_hessian.data(), dim(), dim());
      negative_hessian = -log_density_hessian;
    }
    curvature_direction_.refresh(
      negative_hessian, metric_scale, opts_.curvature_max_rank,
      opts_.sketch_eigenvalue_cutoff);
  }

  Eigen::VectorXd position_direction_noise_() {
    return metric_scale_().cwiseProduct(
      position_sketch_.transform(normal_rng_(dim()),
                                 normal_rng_(position_sketch_.rank()),
                                 opts_.sketch_residual_fraction));
  }

  Eigen::VectorXd curvature_direction_noise_() {
    return metric_scale_().cwiseProduct(
      curvature_direction_.transform(
        normal_rng_(curvature_direction_.rank())));
  }

  void reset_mixture_() {
    position_sketch_.reset();
    curvature_direction_.reset();
    for (int c = 0; c < kComponents; ++c) {
      mixture_logw_[c] = mixture_prior_(c);
    }
    blocks_.clear();
    bandit_updates_ = 0;
    last_component_ = kDiag;
    refresh_mixture_weights_();
  }

  Eigen::VectorXd direction_noise_() {
    const Eigen::Index D = dim();
    const double alpha = opts_.direction_lowrank_weight;
    const double min_diag_fraction = opts_.direction_min_diag_fraction;
    const Eigen::VectorXd& base_var = diagonal_variance_();
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

  void initialize_pca_schedule_() {
    std::size_t final_start = 0;
    const auto& closures = windowed_adaptation_.closures();
    if (adaptation_span_ > 0) {
      final_start = 1;
      if (closures.size() >= 2 && closures.back() == adaptation_span_) {
        final_start = closures[closures.size() - 2] + 1;
      }
    }

    const std::size_t final_length = final_start <= adaptation_span_ ?
      adaptation_span_ - final_start + 1 : 0;
    // Zero-valued PCA options intentionally disable low-rank calibration.
    pca_calibration_enabled_ =
      opts_.J > 0 && opts_.pca_freeze_fraction > 0.0 && final_length > 2;
    if (!pca_calibration_enabled_) {
      pca_freeze_draw_ = adaptation_span_;
      return;
    }

    const auto tail = static_cast<std::size_t>(
      std::ceil(opts_.pca_freeze_fraction * final_length));
    const std::size_t tail_length =
      std::clamp<std::size_t>(tail, 1, final_length);
    pca_freeze_draw_ =
      std::max(final_start, adaptation_span_ - tail_length);
  }

  bool set_projection_basis_from_estimator_() {
    if (online_pca_.count() < opts_.J) {
      return false;
    }
    return set_projection_basis_(online_pca_.vectors());
  }

  void update_basis_estimator_(const Eigen::VectorXd& centered) {
    online_pca_.update(centered);
  }

  // The center OnlinePCA's states are taken relative to; see
  void reset_basis_estimator_() {
    online_pca_.reset();
  }

  void reset_pca_estimator_() {
    online_pca_.reset();
  }

  bool set_projection_basis_(Eigen::MatrixXd basis) {
    if (opts_.J <= 0 || basis.cols() < opts_.J || !basis.allFinite()) {
      projection_basis_ready_ = false;
      return false;
    }
    basis = basis.leftCols(opts_.J).eval();
    for (Eigen::Index j = 0; j < basis.cols(); ++j) {
      const double norm = basis.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        projection_basis_ready_ = false;
        return false;
      }
      basis.col(j) /= norm;
    }
    projection_basis_ = std::move(basis);
    projection_basis_ready_ = true;
    return true;
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
    lowrank_ready_ = projected_pair_count_ >= opts_.lowrank_min_activations;
  }

  void freeze_pca_for_final_calibration_() {
    if (pca_frozen_) {
      return;
    }

    // Bank the variances accumulated against the outgoing basis before
    // installing a new one, otherwise a whole window's worth of projected
    // moments is discarded in favour of the short calibration tail.
    activate_projected_pair_();

    const bool frozen = set_projection_basis_from_estimator_();
    if (!frozen && projected_pair_count_ > 0) {
      projection_basis_ = eigvecs_.leftCols(opts_.J);
      projection_basis_ready_ = projection_basis_.allFinite();
    }

    projected_moments_.reset();
    reset_pca_estimator_();
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
      set_covariance_(handoff.covariance);
      // Diagnostic only: the sampler reads cov_, never transport_cov_. It is
      // kept so the handoff estimate can be compared against the diagonal the
      // windows go on to produce, which examples/example.cpp prints beside
      // metric_variance(). Nothing else would notice its removal, and the
      // `requires` check there means the build would not either.
      transport_cov_ = handoff.covariance;
    }
    seed_mean_from_handoff_(handoff);
    // The transport's principal directions seed the low-rank basis, so the
    // projected moments start accumulating against it from the first
    // post-transport draw.
    if (!handoff.pca_ready || !set_projection_basis_(handoff.pca_basis)) {
      projection_basis_ready_ = false;
    }
    reset_basis_estimator_();
    initialize_K_window_metric_(false);
  }

  // adaptation_draw counts from the end of the transport phase and is zero
  // outside post-transport warmup.
  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    // Past the adapting span lies the terminal buffer, over which every slow
    // and fast parameter stays frozen so the chain equilibrates to the kernel
    // that will actually sample.
    if (!adapting_(adaptation_draw)) {
      return;
    }

    if (pca_calibration_enabled_ && !pca_frozen_ &&
        adaptation_draw >= pca_freeze_draw_) {
      freeze_pca_for_final_calibration_();
    }

    online_moments_.update(theta);
    if (opts_.mixture_direction) {
      // Position accumulation is streaming. Curvature instead probes the
      // local score operator only when this window closes below.
      if (opts_.position_sketch_direction) {
        position_sketch_.update(theta);
      }
    } else if (pca_calibration_enabled_) {
      update_projected_moments_(theta);
      if (!pca_frozen_) {
        update_basis_estimator_(theta - mean_);
      }
    }

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      set_covariance_(online_moments_.variance());
      online_moments_.reset();

      // Slow parameters change only here; the bandit's block is closed too,
      // since its rewards were measured against the outgoing diagonal.
      if (opts_.mixture_direction) {
        finalize_bandit_block_();
        rebuild_mixture_();
      }

      if (pca_calibration_enabled_ && !pca_frozen_) {
        activate_projected_pair_();
        const bool has_next_basis = set_projection_basis_from_estimator_();
        projected_moments_.reset();
        if (!has_next_basis) {
          projection_basis_ready_ = false;
        }
        reset_pca_estimator_();
      }
    }

    if (pca_calibration_enabled_ && pca_frozen_ &&
        adaptation_draw == adaptation_span_) {
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

  // Ordered overrelaxation given a position u_raw under the fitted CDF, with
  // the trip back from a standard normal draw to the line coordinate supplied
  // by the caller.
  //
  // Every branch here has to be reversible with respect to q, because kl_step_
  // cancels the proposal kernel out of the Hastings ratio on exactly that
  // assumption.
  template <typename ToCoordinate>
  double overrelaxed_proposal_from_cdf_(const double u_raw,
                                        ToCoordinate to_coordinate) {
    if (numerics::probability_saturated(u_raw)) {
      // `from` lies past the last quantile of q that a double can distinguish,
      // so u_raw says nothing usable about where it is and the overrelaxation
      // kernel would be applied to a clamped surrogate. Fall back to an
      // independence draw from q, which is exactly reversible for any K.
      //
      // Selecting the fallback on a state-dependent condition does break
      // detailed balance in principle -- from a saturated t0 the kernel is
      // q(.), while the reverse move from an unsaturated t1 would overrelax --
      // and proposing `from` itself instead was tried on exactly that
      // reasoning. It is worse on both counts. The imbalance is not repaired:
      // the identity sends zero mass out of the saturated set while
      // overrelaxation still sends mass in, so the set becomes absorbing, and
      // the flux mismatch stays O(q(saturated)) either way -- about 1e-16,
      // since the set is |z| > 8.15 under q. What the identity does change is
      // that a chain reaching that set can no longer leave, and on earnings
      // that turned a 1/40 failure-to-reach-the-posterior rate into 3/40:
      // seeds 2 (both samplers) and 18 froze near the initialisation point for
      // all 15000 sampling draws. The independence draw is the escape hatch,
      // and it is what makes the saturated branch a rescue rather than a trap.
      return to_coordinate(
        normal_quantile(clamp_probability_(std_uniform_(rng_))));
    }
    const double up = overrelaxed_proposal_impl_(clamp_probability_(u_raw));
    return to_coordinate(normal_quantile(clamp_probability_(up)));
  }

  double overrelaxed_proposal_impl_(const double u) {
    // K = 0 is the independence kernel. The choice is a fixed property of the
    // configuration rather than of the current state, so it is q-reversible.
    if (opts_.K == 0) {
      return clamp_probability_(std_uniform_(rng_));
    }

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
