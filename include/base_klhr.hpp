#pragma once

#include "bfgs.hpp"
#include "gausshermite.hpp"
#include "klhr_numerics.hpp"
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

namespace klhr {

inline constexpr double positive_infinity =
  std::numeric_limits<double>::infinity();

struct KlhrOptions {
  std::uint64_t seed = 0;
  Eigen::Index N = 8;
  double tol = 1e-10;
  double grad_clip = positive_infinity;
  double sas_arg_clip = 30;
  double gtol = 1e-3;
  std::size_t K = 16;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  Eigen::Index J = 1;
  Eigen::Index direction_noise_rank = 1;
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
  double transport_max_segment_logp_drop = 1000.0;
  double transport_max_endpoint_from_best_drop = 100.0;
  double transport_direction_persistence = 0.9;
  double transport_failure_direction_decay = 0.25;
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
    online_moments_(bsm_.dim()),
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

  Eigen::VectorXd draw() {
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      auto result = transport_.step(
        bsm_, rng_, std_uniform_, std_normal_,
        [this](const Eigen::VectorXd& theta) {
          return bsm_.param_constrain(theta);
        });
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

    Eigen::VectorXd rho = random_direction();
    kl_step_(rho);
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
    rho /= norm + opts_.tol;

    return rho;
  }

protected:
  virtual Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                                    const Eigen::VectorXd& rho) = 0;

  virtual double overrelaxed_proposal_(const Eigen::VectorXd& eta) = 0;

  virtual double transition_density_(const double from, const double to,
                                     const Eigen::VectorXd& eta) const = 0;

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
    bool hessian_usable = false;
    bool hessian_identity = false;
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

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init);
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
      const double identity_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(inverse_hessian));
      out.hessian_identity =
        std::abs(inverse_hessian - 1.0) <= identity_tolerance;
    }
    out.success = mode.success;
    return out;
  }

  template <typename EvaluateKl, typename TransformParameters>
  Eigen::VectorXd fit_line_with_kl_fallback_(
      const Eigen::VectorXd& center,
      const Eigen::VectorXd& rho,
      const Eigen::Index parameter_count,
      EvaluateKl evaluate_kl,
      TransformParameters transform_parameters) {
    const LineModeEstimate mode = fit_line_mode_(center, rho);
    if (mode.success && mode.hessian_usable && !mode.hessian_identity) {
      Eigen::VectorXd eta = Eigen::VectorXd::Zero(parameter_count);
      eta(0) = mode.mode;
      eta(1) = mode.log_scale;
      return eta;
    }

    Eigen::VectorXd init = Eigen::VectorXd::Zero(parameter_count);
    init(0) = mode.mode;

    auto kl = [&](const Eigen::VectorXd& eta,
                  double& value, Eigen::VectorXd& grad) {
      evaluate_kl(eta, mode.log_scale, value, grad);
    };

    bfgs::BfgsResult fit = bfgs::bfgs(kl, init);
    nfev_ += fit.nfev * opts_.N;
    const Eigen::VectorXd raw =
      fit.x.size() == parameter_count && fit.x.allFinite() ? fit.x : init;
    return transform_parameters(raw, mode.log_scale);
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
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    if (!std::isfinite(options.l)) {
      options.l = 0.0;
    }
    options.J = std::clamp(options.J, Eigen::Index{0}, D);
    if (options.direction_noise_rank < 0) {
      options.direction_noise_rank = options.J;
    } else {
      options.direction_noise_rank =
        std::clamp(options.direction_noise_rank, Eigen::Index{0}, options.J);
    }
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

  static ReflectedTransportOptions transport_options_(
      const KlhrOptions& options) {
    return {
      .quadrature_size = options.N,
      .pca_rank = options.J,
      .tol = options.tol,
      .grad_clip = options.grad_clip,
      .gtol = options.gtol,
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
    };
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  ReflectedTransport transport_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  OnlinePCA online_pca_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
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
  std::size_t pca_freeze_draw_ = 0;
  std::size_t projected_pair_count_ = 0;

  std::size_t draw_;

  void kl_step_(const Eigen::VectorXd& rho) {
    auto update_acceptance = [this](const bool accepted) {
      const double d = accepted - acceptance_rate_;
      acceptance_rate_ += d / draw_;
    };

    const Eigen::VectorXd eta = fit_line_(theta_, rho);
    const double xi = overrelaxed_proposal_(eta);
    const Eigen::VectorXd thetap = xi * rho + theta_;
    if (!thetap.allFinite()) {
      update_acceptance(false);
      return;
    }

    const double ldp = bsm_.log_density_noe(thetap);
    ++nfev_;
    if (!std::isfinite(ldp)) {
      update_acceptance(false);
      return;
    }

    const double f = transition_density_(0.0, xi, eta);
    if (!std::isfinite(f)) {
      update_acceptance(false);
      return;
    }

    const Eigen::VectorXd reta = fit_line_(thetap, rho);
    const double r = transition_density_(0.0, -xi, reta);
    const double a = ldp - log_density_ + r - f;
    const double log_u = std::log(std_uniform_(rng_));
    const bool accepted =
      std::isfinite(a) && (a >= 0.0 || log_u < a);
    update_acceptance(accepted);
    if (accepted) {
      theta_ = thetap;
      log_density_ = ldp;
    }
  }

  void reset_adaptation_to_defaults_(const bool preserve_window_position) {
    const Eigen::Index D = dim();
    mean_ = Eigen::VectorXd::Zero(D);
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
    const Eigen::Index rank = direction_noise_rank_();
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

  Eigen::Index direction_noise_rank_() const {
    return lowrank_ready_ ? opts_.direction_noise_rank : 0;
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
      opts_.J > 0 && opts_.direction_noise_rank > 0 &&
      opts_.pca_freeze_fraction > 0.0 && final_length > 2;
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

  void apply_transport_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    theta_ = handoff.state.theta;
    log_density_ = handoff.state.log_density;
    if (handoff.rollback) {
      reset_adaptation_to_defaults_(true);
      return;
    }

    if (handoff.covariance.allFinite()) {
      cov_ = handoff.covariance;
    }
    if (!handoff.pca_ready ||
        !set_mean_direction_(handoff.pca_basis, handoff.pca_weights,
                             handoff.pca_whitened)) {
      mean_direction_basis_.setZero();
      mean_direction_weights_.setOnes();
      mean_direction_ready_ = false;
      mean_direction_whitened_ = false;
    }
    online_pca_.reset();
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

  double overrelaxed_proposal_impl_(double u) {
    u = clamp_probability_(u);
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

  double overrelaxed_density_(double from, double to) const {
    from = clamp_probability_(from);
    to = clamp_probability_(to);
    if (opts_.K == 0) {
      return 0.0;
    }
    if (from == to) {
      return -positive_infinity;
    }

    const int K = opts_.K;
    double log_density = -positive_infinity;
    if (to < from) {
      const double log_from = std::log(from);
      const double v = to / from;
      for (int r = K / 2 + 1; r <= K; ++r) {
        const double a = K - r + 1.0;
        const double b = 2.0 * r - K;
        const double term =
          binomial_density_(K, r, from) + beta_density_(v, a, b) -
          log_from;
        log_density = log_sum_exp_(log_density, term);
      }
    } else {
      const double log_one_minus_from = std::log1p(-from);
      const double v = (1.0 - to) / (1.0 - from);
      for (int r = 0; r < (K + 1) / 2; ++r) {
        const double a = r + 1.0;
        const double b = K - 2.0 * r;
        const double term =
          binomial_density_(K, r, from) + beta_density_(v, a, b) -
          log_one_minus_from;
        log_density = log_sum_exp_(log_density, term);
      }
    }
    return log_density;
  }

  static double binomial_density_(int n, int k, double p) {
    p = clamp_probability_(p);
    return std::lgamma(n + 1.0) -
      std::lgamma(k + 1.0) -
      std::lgamma(n - k + 1.0) +
      k * std::log(p) +
      (n - k) * std::log1p(-p);
  }

  static double beta_density_(double x, double a, double b) {
    if (!(x > 0.0 && x < 1.0) || !(a > 0.0 && b > 0.0)) {
      return -positive_infinity;
    }
    return (a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) +
      std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
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
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

  static double log_sum_exp_(double a, double b) {
    if (!std::isfinite(a)) {
      return b;
    }
    if (!std::isfinite(b)) {
      return a;
    }
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
  }

};

} // namespace klhr
