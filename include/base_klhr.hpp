#pragma once

#include "bfgs.hpp"
#include "gausshermite.hpp"
#include "gausslaguerre.hpp"
#include "onlinepca.hpp"

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

struct KlhrOptions {
  std::uint64_t seed = 0;
  Eigen::Index N = 8;
  double tol = 1e-10;
  double grad_clip = std::numeric_limits<double>::infinity();
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
    windowed_adaptation_(opts_.warmup, opts_.windowsize,
                         opts_.windowscale),
    online_moments_(bsm_.dim()),
    transport_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), opts_.J, opts_.l, opts_.tol),
    projected_moments_(opts_.J),
    transport_distance_(opts_.transport_initial_distance) {

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
    gauss_laguerre(opts_.N, laguerre_w_, laguerre_x_);

    const Eigen::Index D = dim();
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J + 1);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J + 1);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_weights_ = Eigen::VectorXd::Ones(opts_.J);
    transport_handoff_pca_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    transport_handoff_pca_weights_ = Eigen::VectorXd::Zero(opts_.J);
    transport_mean_smooth_ = theta_;
    transport_cov_smooth_ = Eigen::VectorXd::Ones(D);
    transport_moments_.update(theta_);
    grad_ = Eigen::VectorXd::Zero(D);
    transport_direction_state_ = normal_rng_(D);
    regularize_transport_direction_();

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    bsm_.log_density_gradient_noe(theta_, log_density_, grad_);
    ++nfev_;
    transport_initial_theta_ = theta_;
    transport_initial_grad_ = grad_;
    transport_initial_log_density_ = log_density_;
    transport_best_theta_ = theta_;
    transport_best_grad_ = grad_;
    transport_best_log_density_ = log_density_;
    draw_ = 0;

    if (!(transport_distance_ > opts_.transport_min_distance) ||
        !std::isfinite(transport_distance_)) {
      transport_distance_ = 1.0;
    }
  }

  virtual ~BaseKLHR() = default;

  std::size_t dim() {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      initial_transport_step_();
      adapt_transport_warmup_(draw_);
      if (draw_ == opts_.initial_transport_steps) {
        finalize_transport_handoff_();
      }
      return bsm_.param_constrain(theta_);
    }

    Eigen::VectorXd rho = random_direction();
    regular_kl_step_(rho);
    record_transport_step_(nan_(), nan_(), nan_(), false, false,
                           missing_unconstrained_draw_());
    adapt_warmup_(theta_, draw_);
    return bsm_.param_constrain(theta_);
  }

  const std::vector<Eigen::VectorXd>& proposal_draw_history() const {
    return proposal_draws_;
  }

  const std::vector<double>& proposal_log_accept_history() const {
    return proposal_log_accept_;
  }

  const std::vector<double>& proposal_log_density_history() const {
    return proposal_log_density_;
  }

  const std::vector<double>& proposal_accepted_history() const {
    return proposal_accepted_;
  }

  const std::vector<double>& proposal_valid_history() const {
    return proposal_valid_;
  }

  const std::vector<double>& transport_distance_history() const {
    return transport_distance_history_;
  }

  const std::vector<double>& transport_reflections_history() const {
    return transport_reflections_;
  }

  const std::vector<double>& transport_logp_gain_history() const {
    return transport_logp_gain_;
  }

  const std::vector<double>& transport_uturn_history() const {
    return transport_uturn_;
  }

  const std::vector<double>& transport_moved_history() const {
    return transport_moved_;
  }

  const std::vector<Eigen::VectorXd>& transport_variance_history() const {
    return transport_variance_;
  }

  const std::vector<double>& transport_direction_norm_history() const {
    return transport_direction_norm_;
  }

  const Eigen::MatrixXd& transport_handoff_pca_basis() const {
    return transport_handoff_pca_basis_;
  }

  const Eigen::VectorXd& transport_handoff_pca_weights() const {
    return transport_handoff_pca_weights_;
  }

  std::size_t transport_handoff_pca_count() const {
    return transport_handoff_pca_count_;
  }

  bool transport_handoff_pca_ready() const {
    return transport_handoff_pca_ready_;
  }

  bool transport_handoff_pca_whitened() const {
    return transport_handoff_pca_whitened_;
  }

  bool transport_rollback() const {
    return transport_rollback_;
  }

  double transport_initial_log_density() const {
    return transport_initial_log_density_;
  }

  double transport_best_log_density() const {
    return transport_best_log_density_;
  }

  double transport_endpoint_from_best_drop() const {
    return transport_endpoint_from_best_drop_;
  }

  Eigen::VectorXd random_direction() {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
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
                                     const Eigen::VectorXd& eta) = 0;

  virtual Eigen::VectorXd fit_transport_ray_(const Eigen::VectorXd& center,
                                             const Eigen::VectorXd& rho) {
    double distance0 = transport_distance_;
    if (!(distance0 > opts_.transport_min_distance) ||
        !std::isfinite(distance0)) {
      distance0 = opts_.transport_initial_distance;
    }
    if (!(distance0 > opts_.transport_min_distance) ||
        !std::isfinite(distance0)) {
      distance0 = 1.0;
    }
    distance0 = std::clamp(distance0, opts_.transport_min_distance,
                           opts_.transport_max_distance);
    const double log_scale0 = std::log(distance0);

    Eigen::VectorXd init(2);
    init << 0.0, 0.0;
    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      transport_weibull_KL_(eta, center, rho, log_scale0, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 4});
    nfev_ += o.nfev * static_cast<std::size_t>(laguerre_x_.size());

    const Eigen::VectorXd raw =
      o.x.size() == 2 && o.x.allFinite() ? o.x : init;
    Eigen::VectorXd out(2);
    out << bounded_log_shape_(raw(0)),
      relative_log_scale_(raw(1), log_scale0);
    return out;
  }

  virtual double transport_distance_proposal_(const Eigen::VectorXd& eta) {
    if (eta.size() < 2 || !eta.allFinite()) {
      return nan_();
    }
    auto [shape, scale] = unpack_weibull_(eta);
    if (!(shape > 0.0) || !(scale > 0.0) ||
        !std::isfinite(shape) || !std::isfinite(scale)) {
      return nan_();
    }
    const double u = clamp_probability_(std_uniform_(rng_));
    const double x = -std::log1p(-u);
    const double distance = scale * std::pow(x, 1.0 / shape);
    if (!(distance > 0.0) || !std::isfinite(distance)) {
      return nan_();
    }
    return distance;
  }

  virtual void record_kl_step_(const Eigen::VectorXd& eta, const double xi,
                               const bool accepted) {
    (void) eta;
    (void) xi;
    (void) accepted;
  }

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
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
    out.mode = mode.x(0);
    const double h = mode.hess_inv(0, 0);
    if (std::isfinite(h) && h > 0.0) {
      out.log_scale = 0.5 * std::log(h * 1.1);
    }
    return out;
  }

  static KlhrOptions normalized_options_(KlhrOptions options,
                                         const std::size_t dim) {
    const Eigen::Index D = static_cast<Eigen::Index>(dim);
    options.J = std::clamp(options.J, Eigen::Index{0}, D);
    if (options.direction_noise_rank < 0) {
      options.direction_noise_rank = options.J;
    } else {
      options.direction_noise_rank =
        std::clamp(options.direction_noise_rank, Eigen::Index{0}, options.J);
    }
    options.pca_freeze_fraction =
      std::clamp(options.pca_freeze_fraction, 0.0, 1.0);
    options.transport_cov_shrink =
      std::clamp(options.transport_cov_shrink, 0.0, 1.0);
    options.transport_cov_ratio_cap =
      std::max(options.transport_cov_ratio_cap, 1.0);
    if (std::isfinite(options.transport_max_endpoint_from_best_drop)) {
      options.transport_max_endpoint_from_best_drop =
        std::max(0.0, options.transport_max_endpoint_from_best_drop);
    } else {
      options.transport_max_endpoint_from_best_drop =
        std::numeric_limits<double>::infinity();
    }
    return options;
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  mcmcpp::WelfordAccumulator transport_moments_;
  OnlinePCA online_pca_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd laguerre_x_; // Gauss-Laguerre sample points
  Eigen::VectorXd laguerre_w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
  Eigen::MatrixXd projection_basis_;
  Eigen::MatrixXd mean_direction_basis_;
  Eigen::VectorXd mean_direction_weights_;
  Eigen::MatrixXd transport_handoff_pca_basis_;
  Eigen::VectorXd transport_handoff_pca_weights_;
  Eigen::VectorXd transport_mean_smooth_;
  Eigen::VectorXd transport_cov_smooth_;
  std::vector<Eigen::VectorXd> proposal_draws_;
  std::vector<double> proposal_log_accept_;
  std::vector<double> proposal_log_density_;
  std::vector<double> proposal_accepted_;
  std::vector<double> proposal_valid_;
  std::vector<double> transport_distance_history_;
  std::vector<double> transport_reflections_;
  std::vector<double> transport_logp_gain_;
  std::vector<double> transport_uturn_;
  std::vector<double> transport_moved_;
  std::vector<double> transport_direction_norm_;
  std::vector<Eigen::VectorXd> transport_variance_;
  Eigen::VectorXd grad_;
  Eigen::VectorXd transport_direction_state_;
  Eigen::VectorXd transport_initial_theta_;
  Eigen::VectorXd transport_initial_grad_;
  Eigen::VectorXd transport_best_theta_;
  Eigen::VectorXd transport_best_grad_;
  double transport_distance_;
  double transport_initial_log_density_ = std::numeric_limits<double>::quiet_NaN();
  double transport_best_log_density_ = std::numeric_limits<double>::quiet_NaN();
  double transport_endpoint_from_best_drop_ =
    std::numeric_limits<double>::quiet_NaN();
  bool projection_basis_ready_ = false;
  bool mean_direction_ready_ = false;
  bool mean_direction_whitened_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  bool transport_rollback_ = false;
  std::size_t projected_pair_count_ = 0;
  std::size_t transport_handoff_pca_count_ = 0;
  bool transport_handoff_pca_ready_ = false;
  bool transport_handoff_pca_whitened_ = false;

  std::size_t draw_;

  void regular_kl_step_(const Eigen::VectorXd& rho) {
    const double missing_xi = std::numeric_limits<double>::quiet_NaN();
    const Eigen::VectorXd missing_draw = missing_constrained_draw_();
    auto update_acceptance = [this](const bool accepted) {
      const double d = static_cast<double>(accepted) - acceptance_rate_;
      acceptance_rate_ += d / draw_;
    };
    auto reject = [this, &update_acceptance](
                    const Eigen::VectorXd& eta,
                    const double xi,
                    const Eigen::VectorXd& proposal,
                    const double proposal_log_density,
                    const double log_accept,
                    const bool valid) {
      update_acceptance(false);
      record_proposal_step_(proposal, proposal_log_density, log_accept,
                            false, valid);
      record_kl_step_(eta, xi, false);
    };

    Eigen::VectorXd eta = fit_line_(theta_, rho);
    if (!eta.allFinite()) {
      reject(eta, missing_xi, missing_draw,
             missing_xi, missing_xi, false);
      return;
    }

    const double xi = overrelaxed_proposal_(eta);
    if (!std::isfinite(xi)) {
      reject(eta, xi, missing_draw,
             missing_xi, missing_xi, false);
      return;
    }

    Eigen::VectorXd thetap = xi * rho + theta_;
    if (!thetap.allFinite()) {
      reject(eta, xi, missing_draw,
             missing_xi, missing_xi, false);
      return;
    }
    const Eigen::VectorXd proposal = constrain_or_missing_(thetap);

    double ldp = bsm_.log_density_noe(thetap);
    ++nfev_;
    if (!std::isfinite(ldp)) {
      reject(eta, xi, proposal,
             ldp, missing_xi, false);
      return;
    }

    const double f = transition_density_(0.0, xi, eta);
    if (!std::isfinite(f)) {
      reject(eta, xi, proposal,
             ldp, missing_xi, false);
      return;
    }

    Eigen::VectorXd reta = fit_line_(thetap, rho);
    if (!reta.allFinite()) {
      reject(eta, xi, proposal,
             ldp, missing_xi, false);
      return;
    }

    const double r = transition_density_(0.0, -xi, reta);
    if (!std::isfinite(r)) {
      reject(eta, xi, proposal,
             ldp, missing_xi, false);
      return;
    }

    double a = ldp - log_density_ + r - f;
    if (!std::isfinite(a)) {
      reject(eta, xi, proposal,
             ldp, a, false);
      return;
    }

    const bool accepted = std::log(std_uniform_(rng_)) < std::min(0.0, a);
    update_acceptance(accepted);
    record_proposal_step_(proposal, ldp, a, accepted, true);
    record_kl_step_(eta, xi, accepted);
    if (accepted) {
      theta_ = thetap;
      log_density_ = ldp;
    }
  }

  struct TransportState {
    Eigen::VectorXd theta;
    Eigen::VectorXd grad;
    double log_density = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
  };

  void initial_transport_step_() {
    auto update_acceptance = [this](const bool moved) {
      const double d = static_cast<double>(moved) - acceptance_rate_;
      acceptance_rate_ += d / draw_;
    };

    const Eigen::VectorXd theta0 = theta_;
    const double logp0 = log_density_;
    const Eigen::VectorXd grad0 = grad_;
    const Eigen::VectorXd scale = metric_scale_();
    partial_refresh_transport_direction_();
    const Eigen::VectorXd direction0 = normalized_transport_direction_();
    const Eigen::VectorXd missing_draw = missing_constrained_draw_();
    const Eigen::VectorXd missing_eta =
      Eigen::VectorXd::Constant(3, std::numeric_limits<double>::quiet_NaN());

    if (!theta0.allFinite() || !grad0.allFinite() || !scale.allFinite() ||
        !std::isfinite(logp0) || !direction0.allFinite()) {
      update_acceptance(false);
      regularize_transport_direction_();
      record_proposal_step_(missing_draw, nan_(), nan_(), false, false);
      record_kl_step_(missing_eta, nan_(), false);
      record_transport_step_(nan_(), 0.0, nan_(), false, false,
                             missing_unconstrained_draw_());
      return;
    }

    TransportState current;
    current.theta = theta0;
    current.grad = grad0;
    current.log_density = logp0;
    current.valid = true;

    TransportState endpoint = current;
    Eigen::VectorXd direction = direction0;
    Eigen::VectorXd endpoint_direction = direction0;
    bool moved = false;
    bool uturn = false;
    bool failed = false;
    double total_distance = 0.0;
    double endpoint_logp_gain = nan_();
    std::size_t reflections = 0;

    for (std::size_t step = 0; step < opts_.transport_max_reflections; ++step) {
      const Eigen::VectorXd rho = (scale.array() * direction.array()).matrix();
      Eigen::VectorXd eta = fit_transport_ray_(current.theta, rho);
      double distance = transport_distance_proposal_(eta);
      if (!eta.allFinite() || !(distance > 0.0) || !std::isfinite(distance)) {
        failed = true;
        break;
      }
      distance = std::clamp(distance, opts_.transport_min_distance,
                            opts_.transport_max_distance);

      TransportState next = transport_ray_step_(current, distance, rho);
      if (!next.valid) {
        failed = true;
        break;
      }

      if (logp0 - next.log_density > std::abs(opts_.transport_max_logp_drop)) {
        failed = true;
        break;
      }

      if (current.log_density - next.log_density >
          std::abs(opts_.transport_max_segment_logp_drop)) {
        break;
      }

      const Eigen::VectorXd delta = next.theta - theta0;
      if (!delta.allFinite()) {
        failed = true;
        break;
      }
      const double turning = delta.dot(direction);
      const double initial_turning = delta.dot(direction0);
      if (!std::isfinite(turning) || !std::isfinite(initial_turning) ||
          turning <= 0.0 || initial_turning <= 0.0) {
        uturn = true;
        break;
      }

      endpoint = next;
      current = next;
      moved = true;
      ++reflections;
      total_distance += distance;
      transport_distance_ = distance;
      endpoint_logp_gain = endpoint.log_density - logp0;

      const Eigen::VectorXd reflected =
        reflected_transport_direction_(direction, endpoint.grad, scale);
      if (!reflected.allFinite()) {
        break;
      }
      direction = reflected;
      endpoint_direction = direction;
    }

    Eigen::VectorXd proposal = missing_draw;
    double proposal_log_density = nan_();
    if (moved) {
      theta_ = endpoint.theta;
      grad_ = endpoint.grad;
      log_density_ = endpoint.log_density;
      proposal = constrain_or_missing_(endpoint.theta);
      proposal_log_density = endpoint.log_density;
      if (!proposal.allFinite()) {
        moved = false;
        theta_ = theta0;
        grad_ = grad0;
        log_density_ = logp0;
        proposal_log_density = nan_();
      } else {
        update_transport_covariance_(theta_);
        update_best_transport_state_();
      }
    }

    update_acceptance(moved);
    update_transport_direction_(moved, failed, endpoint_direction, direction0);
    record_proposal_step_(proposal, proposal_log_density, nan_(),
                          moved, moved);
    record_kl_step_(missing_eta, nan_(), false);
    record_transport_step_(total_distance, static_cast<double>(reflections),
                           endpoint_logp_gain, uturn, moved,
                           scale.array().square().matrix());
  }

  TransportState transport_ray_step_(const TransportState& state,
                                     const double distance,
                                     const Eigen::VectorXd& rho) {
    TransportState out;
    out.theta = Eigen::VectorXd::Constant(dim(), nan_());
    out.grad = Eigen::VectorXd::Constant(dim(), nan_());
    if (!state.valid || !(distance > 0.0) || !std::isfinite(distance) ||
        !state.theta.allFinite() || !state.grad.allFinite() ||
        !rho.allFinite()) {
      return out;
    }

    out.theta = state.theta + distance * rho;
    if (!out.theta.allFinite()) {
      return out;
    }

    bsm_.log_density_gradient_noe(out.theta, out.log_density, out.grad);
    ++nfev_;
    if (!std::isfinite(out.log_density) || !out.grad.allFinite()) {
      return out;
    }

    out.valid = true;
    return out;
  }

  void transport_weibull_KL_(const Eigen::VectorXd& eta,
                             const Eigen::VectorXd& center,
                             const Eigen::VectorXd& rho,
                             const double log_scale0,
                             double& value,
                             Eigen::VectorXd& grad) {
    const double log_shape = bounded_log_shape_(eta(0));
    const double dlog_shape = bounded_log_shape_derivative_(eta(0));
    const double shape = scale_from_log_(log_shape);
    const double log_scale = relative_log_scale_(eta(1), log_scale0);
    const double dlog_scale = relative_log_scale_derivative_(eta(1));
    const double scale = scale_from_log_(log_scale);
    if (!std::isfinite(log_shape) || !std::isfinite(dlog_shape) ||
        !std::isfinite(shape) || !std::isfinite(log_scale) ||
        !std::isfinite(dlog_scale) || !std::isfinite(scale) ||
        !center.allFinite() || !rho.allFinite()) {
      set_bad_kl_(eta, value, grad);
      return;
    }

    value = 0.0;
    grad = Eigen::VectorXd::Zero(2);

    Eigen::Index D = dim();
    Eigen::VectorXd xi(D);
    Eigen::VectorXd grad_logp(D);
    const double inv_shape = 1.0 / shape;
    for (Eigen::Index n = 0; n < laguerre_x_.size(); ++n) {
      const double xn = laguerre_x_(n);
      const double wn = laguerre_w_(n);
      if (!(xn > 0.0) || !std::isfinite(xn) || !std::isfinite(wn)) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      const double log_x = std::log(xn);
      const double x_power =
        std::exp(std::clamp(inv_shape * log_x, -700.0, 700.0));
      const double distance = scale * x_power;
      xi = center + distance * rho;
      if (!(distance > 0.0) || !std::isfinite(distance) || !xi.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      double logp;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      if (!std::isfinite(logp) || !grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      if (!grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      const double line_grad = grad_logp.dot(rho);
      if (!std::isfinite(line_grad)) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      const double log_q =
        log_shape - log_scale + (1.0 - inv_shape) * log_x - xn;
      value += wn * (log_q - logp);

      const double d_distance_d_log_shape = -distance * log_x * inv_shape;
      const double d_distance_d_log_scale = distance;
      grad(0) += wn * (1.0 + inv_shape * log_x -
                       line_grad * d_distance_d_log_shape);
      grad(1) += wn * (-1.0 - line_grad * d_distance_d_log_scale);
    }

    grad(0) *= dlog_shape;
    grad(1) *= dlog_scale;
  }

  Eigen::VectorXd reflected_transport_direction_(
      const Eigen::VectorXd& direction,
      const Eigen::VectorXd& grad,
      const Eigen::VectorXd& scale) {
    if (!direction.allFinite() || !grad.allFinite() || !scale.allFinite()) {
      return missing_unconstrained_draw_();
    }
    Eigen::VectorXd normal = (scale.array() * grad.array()).matrix();
    const double normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= opts_.tol) {
      return missing_unconstrained_draw_();
    }
    normal /= normal_norm;
    Eigen::VectorXd reflected = direction - 2.0 * direction.dot(normal) * normal;
    const double reflected_norm = reflected.norm();
    if (!std::isfinite(reflected_norm) || reflected_norm <= opts_.tol) {
      return missing_unconstrained_draw_();
    }
    return reflected / reflected_norm;
  }

  void partial_refresh_transport_direction_() {
    const double a =
      std::clamp(opts_.transport_direction_persistence, 0.0, 1.0);
    const double b = std::sqrt(std::max(0.0, 1.0 - a * a));
    if (transport_direction_state_.size() != static_cast<Eigen::Index>(dim()) ||
        !transport_direction_state_.allFinite()) {
      transport_direction_state_ = normal_rng_(dim());
    } else {
      transport_direction_state_ = a * transport_direction_state_ + b * normal_rng_(dim());
    }
    regularize_transport_direction_();
  }

  void update_transport_direction_(const bool moved,
                                   const bool failed,
                                   const Eigen::VectorXd& endpoint_direction,
                                   const Eigen::VectorXd& initial_direction) {
    const double decay =
      std::clamp(opts_.transport_failure_direction_decay, 0.0, 1.0);
    if (moved && endpoint_direction.allFinite()) {
      transport_direction_state_ = endpoint_direction;
      if (failed) {
        transport_direction_state_ *= decay;
      }
    } else {
      transport_direction_state_ = -decay * initial_direction;
    }

    regularize_transport_direction_();
  }

  void regularize_transport_direction_() {
    if (transport_direction_state_.size() != static_cast<Eigen::Index>(dim()) ||
        !transport_direction_state_.allFinite()) {
      transport_direction_state_ = normal_rng_(dim());
    }

    double norm = transport_direction_state_.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      transport_direction_state_ = normal_rng_(dim());
    }
  }

  Eigen::VectorXd normalized_transport_direction_() {
    regularize_transport_direction_();
    const double norm = transport_direction_state_.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      return missing_unconstrained_draw_();
    }
    return transport_direction_state_ / norm;
  }

  Eigen::VectorXd missing_constrained_draw_() {
    return Eigen::VectorXd::Constant(dim(),
                                    std::numeric_limits<double>::quiet_NaN());
  }

  Eigen::VectorXd missing_unconstrained_draw_() {
    return Eigen::VectorXd::Constant(dim(),
                                    std::numeric_limits<double>::quiet_NaN());
  }

  Eigen::VectorXd constrain_or_missing_(const Eigen::VectorXd& theta) {
    Eigen::VectorXd out = missing_constrained_draw_();
    if (!theta.allFinite()) {
      return out;
    }
    out = bsm_.param_constrain(theta);
    if (!out.allFinite()) {
      out = missing_constrained_draw_();
    }
    return out;
  }

  void record_proposal_step_(const Eigen::VectorXd& proposal,
                             const double proposal_log_density,
                             const double log_accept,
                             const bool accepted,
                             const bool valid) {
    proposal_draws_.push_back(proposal);
    proposal_log_density_.push_back(proposal_log_density);
    proposal_log_accept_.push_back(log_accept);
    proposal_accepted_.push_back(accepted ? 1.0 : 0.0);
    proposal_valid_.push_back(valid ? 1.0 : 0.0);
  }

  void record_transport_step_(const double distance,
                              const double reflections,
                              const double logp_gain,
                              const bool uturn,
                              const bool moved,
                              const Eigen::VectorXd& variance) {
    transport_distance_history_.push_back(distance);
    transport_reflections_.push_back(reflections);
    transport_logp_gain_.push_back(logp_gain);
    transport_uturn_.push_back(uturn ? 1.0 : 0.0);
    transport_moved_.push_back(moved ? 1.0 : 0.0);
    transport_variance_.push_back(variance);
    transport_direction_norm_.push_back(transport_direction_state_.norm());
  }

  void update_best_transport_state_() {
    if (!theta_.allFinite() || !grad_.allFinite() ||
        !std::isfinite(log_density_)) {
      return;
    }
    if (!std::isfinite(transport_best_log_density_) ||
        log_density_ > transport_best_log_density_) {
      transport_best_theta_ = theta_;
      transport_best_grad_ = grad_;
      transport_best_log_density_ = log_density_;
    }
  }

  void reset_adaptation_to_defaults_(const bool advance_windows_to_current) {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J + 1);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J + 1);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_weights_ = Eigen::VectorXd::Ones(opts_.J);
    transport_handoff_pca_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    transport_handoff_pca_weights_ = Eigen::VectorXd::Zero(opts_.J);
    transport_mean_smooth_ = theta_;
    transport_cov_smooth_ = Eigen::VectorXd::Ones(D);

    online_moments_.reset();
    transport_moments_.reset();
    transport_moments_.update(theta_);
    online_pca_.reset();
    projected_moments_.reset();
    windowed_adaptation_.reset();
    if (advance_windows_to_current) {
      for (std::size_t d = 1; d <= draw_; ++d) {
        (void) windowed_adaptation_.window_closed(d);
      }
    }

    projection_basis_ready_ = false;
    mean_direction_ready_ = false;
    mean_direction_whitened_ = false;
    pca_frozen_ = false;
    lowrank_ready_ = false;
    projected_pair_count_ = 0;
    transport_handoff_pca_count_ = 0;
    transport_handoff_pca_ready_ = false;
    transport_handoff_pca_whitened_ = false;

    transport_distance_ = opts_.transport_initial_distance;
    if (!(transport_distance_ > opts_.transport_min_distance) ||
        !std::isfinite(transport_distance_)) {
      transport_distance_ = 1.0;
    }
    transport_direction_state_ = normal_rng_(D);
    regularize_transport_direction_();
  }

  void rollback_transport_to_initial_state_() {
    if (transport_initial_theta_.size() == static_cast<Eigen::Index>(dim()) &&
        transport_initial_grad_.size() == static_cast<Eigen::Index>(dim()) &&
        transport_initial_theta_.allFinite() &&
        transport_initial_grad_.allFinite() &&
        std::isfinite(transport_initial_log_density_)) {
      theta_ = transport_initial_theta_;
      grad_ = transport_initial_grad_;
      log_density_ = transport_initial_log_density_;
    }
    transport_rollback_ = true;
    reset_adaptation_to_defaults_(true);
  }

  Eigen::VectorXd metric_scale_() {
    Eigen::VectorXd scale(dim());
    for (Eigen::Index d = 0; d < scale.size(); ++d) {
      double v = cov_(d);
      if (!std::isfinite(v) || v <= opts_.tol) {
        v = 1.0;
      }
      scale(d) = std::sqrt(v);
    }
    return scale;
  }

  void update_transport_covariance_(const Eigen::VectorXd& theta) {
    if (theta.size() != static_cast<Eigen::Index>(dim()) ||
        !theta.allFinite()) {
      return;
    }

    transport_moments_.update(theta);
    if (transport_moments_.count() <= 1) {
      return;
    }

    Eigen::VectorXd raw_mean = transport_moments_.mean();
    Eigen::VectorXd raw_var = transport_moments_.variance();
    if (raw_mean.size() != static_cast<Eigen::Index>(dim()) ||
        raw_var.size() != static_cast<Eigen::Index>(dim()) ||
        !raw_mean.allFinite() ||
        !raw_var.allFinite()) {
      return;
    }

    const double shrink = opts_.transport_cov_shrink;
    const double ratio_cap = opts_.transport_cov_ratio_cap;
    if (!(ratio_cap >= 1.0) || !std::isfinite(ratio_cap)) {
      return;
    }

    if (transport_cov_smooth_.size() != raw_var.size() ||
        !transport_cov_smooth_.allFinite()) {
      transport_cov_smooth_ = Eigen::VectorXd::Ones(raw_var.size());
    }

    for (Eigen::Index d = 0; d < raw_var.size(); ++d) {
      const double old_var = std::max(transport_cov_smooth_(d), opts_.tol);
      const double target_var = std::max(raw_var(d), opts_.tol);
      const double proposal =
        (1.0 - shrink) * old_var + shrink * target_var;
      const double lo = old_var / ratio_cap;
      const double hi = old_var * ratio_cap;
      transport_cov_smooth_(d) = std::clamp(proposal, lo, hi);
    }

    if (transport_cov_smooth_.allFinite()) {
      cov_ = transport_cov_smooth_;
    }

    update_transport_mean_smooth_(raw_mean);
    update_transport_mean_direction_basis_(theta);
  }

  void update_transport_mean_smooth_(const Eigen::VectorXd& raw_mean) {
    if (raw_mean.size() != static_cast<Eigen::Index>(dim()) ||
        !raw_mean.allFinite()) {
      return;
    }

    if (transport_mean_smooth_.size() != raw_mean.size() ||
        !transport_mean_smooth_.allFinite()) {
      transport_mean_smooth_ = raw_mean;
      return;
    }

    const double shrink = opts_.transport_cov_shrink;
    const double cap = opts_.transport_cov_ratio_cap;
    for (Eigen::Index d = 0; d < raw_mean.size(); ++d) {
      const double var =
        transport_cov_smooth_.size() == raw_mean.size() ?
        transport_cov_smooth_(d) : 1.0;
      const double sd = std::sqrt(std::max(var, opts_.tol));
      const double max_step = cap * sd;
      const double old_mean = transport_mean_smooth_(d);
      const double proposal =
        old_mean + shrink * (raw_mean(d) - old_mean);
      if (!std::isfinite(proposal) || !std::isfinite(max_step)) {
        continue;
      }
      transport_mean_smooth_(d) =
        std::clamp(proposal, old_mean - max_step, old_mean + max_step);
    }
  }

  void update_transport_mean_direction_basis_(const Eigen::VectorXd& theta) {
    if (opts_.J <= 0 ||
        theta.size() != static_cast<Eigen::Index>(dim()) ||
        !theta.allFinite() ||
        transport_moments_.count() < 2) {
      return;
    }

    if (transport_mean_smooth_.size() != theta.size() ||
        !transport_mean_smooth_.allFinite()) {
      return;
    }

    const Eigen::VectorXd centered = theta - transport_mean_smooth_;
    if (!centered.allFinite() || centered.norm() <= opts_.tol) {
      return;
    }

    online_pca_.update(centered);
    (void) set_mean_direction_from_online_pca_(true);
  }

  Eigen::VectorXd direction_noise_() {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    const double alpha =
      std::clamp(opts_.direction_lowrank_weight, 0.0, 1.0);
    const double min_diag_fraction =
      std::clamp(opts_.direction_min_diag_fraction, 0.0, 1.0);

    Eigen::VectorXd base_var(D);
    Eigen::VectorXd residual_var(D);
    for (Eigen::Index d = 0; d < D; ++d) {
      double v = cov_(d);
      if (!std::isfinite(v) || v <= opts_.tol) {
        v = opts_.tol;
      }
      base_var(d) = v;
      residual_var(d) = v;
    }

    const Eigen::Index rank = direction_noise_rank_();
    if (rank > 0 && alpha > 0.0) {
      for (Eigen::Index k = 0; k < rank; ++k) {
        const double lambda = direction_lowrank_variance_(k);
        const Eigen::VectorXd v = eigvecs_.col(k);
        if (!(lambda > 0.0) || !std::isfinite(lambda) || !v.allFinite()) {
          continue;
        }
        residual_var -= alpha * lambda * v.array().square().matrix();
      }
    }

    for (Eigen::Index d = 0; d < D; ++d) {
      const double floor = std::max(opts_.tol, min_diag_fraction * base_var(d));
      if (!std::isfinite(residual_var(d)) || residual_var(d) < floor) {
        residual_var(d) = floor;
      }
    }

    Eigen::VectorXd noise =
      residual_var.array().sqrt().matrix().cwiseProduct(normal_rng_(D));
    if (rank > 0 && alpha > 0.0) {
      for (Eigen::Index k = 0; k < rank; ++k) {
        const double lambda = direction_lowrank_variance_(k);
        const Eigen::VectorXd v = eigvecs_.col(k);
        if (!(lambda > 0.0) || !std::isfinite(lambda) || !v.allFinite()) {
          continue;
        }
        noise += std::sqrt(alpha * lambda) * std_normal_(rng_) * v;
      }
    }

    if (!noise.allFinite()) {
      return normal_rng_(D);
    }
    return noise;
  }

  Eigen::VectorXd diagonal_direction_noise_() {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    Eigen::VectorXd noise(D);
    for (Eigen::Index d = 0; d < D; ++d) {
      double v = cov_(d);
      if (!std::isfinite(v) || v <= opts_.tol) {
        v = opts_.tol;
      }
      noise(d) = std::sqrt(v) * std_normal_(rng_);
    }
    return noise;
  }

  Eigen::VectorXd mean_direction_noise_() {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    if (!mean_direction_ready_ || opts_.J <= 0 ||
        mean_direction_basis_.rows() != D ||
        mean_direction_basis_.cols() != opts_.J ||
        mean_direction_weights_.size() != opts_.J ||
        !mean_direction_basis_.allFinite() ||
        !mean_direction_weights_.allFinite()) {
      return diagonal_direction_noise_();
    }

    Eigen::VectorXd weights = mean_direction_weights_.cwiseMax(0.0);
    if (!(weights.sum() > opts_.tol) || !weights.allFinite()) {
      weights.setOnes();
    }

    std::discrete_distribution<Eigen::Index> component(
      weights.data(), weights.data() + weights.size());
    const Eigen::Index j = component(rng_);
    if (j < 0 || j >= mean_direction_basis_.cols()) {
      return diagonal_direction_noise_();
    }

    Eigen::VectorXd noise;
    if (mean_direction_whitened_) {
      noise = normal_rng_(D);
      noise += mean_direction_basis_.col(j);
      Eigen::VectorXd scale = metric_scale_();
      if (scale.size() != D || !scale.allFinite()) {
        return diagonal_direction_noise_();
      }
      noise = scale.array() * noise.array();
    } else {
      noise = diagonal_direction_noise_();
      noise += mean_direction_basis_.col(j);
    }
    return noise;
  }

  Eigen::Index direction_noise_rank_() const {
    if (!lowrank_ready_) {
      return 0;
    }
    const Eigen::Index max_rank =
      std::min({opts_.direction_noise_rank, opts_.J,
                eigvecs_.cols(), eigvals_.size()});
    return std::max<Eigen::Index>(0, max_rank);
  }

  double direction_lowrank_variance_(const Eigen::Index k) const {
    if (k < 0 || k >= eigvals_.size() || k >= eigvecs_.cols()) {
      return 0.0;
    }
    const Eigen::VectorXd v = eigvecs_.col(k);
    if (!v.allFinite()) {
      return 0.0;
    }
    double lambda = eigvals_(k);
    if (!std::isfinite(lambda) || lambda <= opts_.tol) {
      return 0.0;
    }
    return lambda;
  }

  bool pca_calibration_enabled_() const {
    return opts_.J > 0 &&
      opts_.direction_noise_rank > 0 &&
      opts_.windowsize > 0 &&
      opts_.pca_freeze_fraction > 0.0 &&
      pca_final_window_length_() > 2;
  }

  std::size_t pca_freeze_start_() const {
    const std::size_t final_start = pca_final_window_start_();
    const std::size_t final_length = pca_final_window_length_();
    const auto tail = static_cast<std::size_t>(
      std::ceil(opts_.pca_freeze_fraction *
                static_cast<double>(final_length)));
    const std::size_t tail_length = std::clamp<std::size_t>(tail, 1,
                                                            final_length);
    return std::max(final_start, opts_.warmup - tail_length);
  }

  std::size_t pca_final_window_length_() const {
    const std::size_t final_start = pca_final_window_start_();
    if (final_start > opts_.warmup) {
      return 0;
    }
    return opts_.warmup - final_start + 1;
  }

  std::size_t pca_final_window_start_() const {
    if (opts_.warmup == 0) {
      return 0;
    }
    if (opts_.windowsize == 0 || opts_.warmup <= opts_.windowsize) {
      return 1;
    }

    const std::size_t scale = std::max<std::size_t>(opts_.windowscale, 1);
    std::size_t window_size = opts_.windowsize;
    std::size_t close_window = opts_.windowsize;
    std::size_t previous_close = 0;

    while (close_window < opts_.warmup) {
      previous_close = close_window;
      if (window_size >
          std::numeric_limits<std::size_t>::max() / scale) {
        break;
      }
      window_size *= scale;

      const std::size_t remaining = opts_.warmup - close_window;
      const bool next_window_reaches_warmup =
        window_size > std::numeric_limits<std::size_t>::max() / scale ||
        scale * window_size >= remaining;
      if (next_window_reaches_warmup) {
        close_window = opts_.warmup;
      } else {
        close_window += window_size;
      }
    }

    return previous_close + 1;
  }

  bool set_projection_basis_from_online_pca_() {
    if (opts_.J <= 0 || online_pca_.count() < static_cast<std::size_t>(opts_.J)) {
      return false;
    }

    Eigen::MatrixXd basis = online_pca_.vectors();
    if (basis.cols() < opts_.J || basis.rows() != static_cast<Eigen::Index>(dim()) ||
        !basis.allFinite()) {
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

  bool set_mean_direction_from_online_pca_(const bool whitened = false) {
    if (opts_.J <= 0 || online_pca_.count() < static_cast<std::size_t>(opts_.J)) {
      return false;
    }

    Eigen::MatrixXd basis = online_pca_.vectors();
    Eigen::VectorXd weights = online_pca_.values();
    if (basis.cols() < opts_.J ||
        basis.rows() != static_cast<Eigen::Index>(dim()) ||
        weights.size() < opts_.J ||
        !basis.allFinite() ||
        !weights.allFinite()) {
      return false;
    }

    mean_direction_basis_ = basis.leftCols(opts_.J);
    mean_direction_weights_ = weights.head(opts_.J).cwiseMax(opts_.tol);
    for (Eigen::Index j = 0; j < mean_direction_basis_.cols(); ++j) {
      const double norm = mean_direction_basis_.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        mean_direction_ready_ = false;
        mean_direction_whitened_ = false;
        return false;
      }
      mean_direction_basis_.col(j) /= norm;
    }
    mean_direction_ready_ = true;
    mean_direction_whitened_ = whitened;
    return true;
  }

  void update_projected_moments_(const Eigen::VectorXd& theta) {
    if (!projection_basis_ready_ || opts_.J <= 0 ||
        projection_basis_.cols() != opts_.J ||
        projection_basis_.rows() != theta.size()) {
      return;
    }

    const Eigen::VectorXd projected = projection_basis_.transpose() * theta;
    if (projected.allFinite()) {
      projected_moments_.update(projected);
    }
  }

  bool activate_projected_pair_() {
    if (!projection_basis_ready_ || opts_.J <= 0 ||
        projected_moments_.count() <= 2) {
      return false;
    }

    Eigen::VectorXd variances = projected_moments_.variance();
    if (variances.size() != opts_.J) {
      return false;
    }
    for (Eigen::Index j = 0; j < variances.size(); ++j) {
      if (!std::isfinite(variances(j)) || variances(j) <= opts_.tol) {
        variances(j) = opts_.tol;
      }
    }

    eigvecs_.leftCols(opts_.J) = projection_basis_;
    eigvals_.head(opts_.J) = variances;
    ++projected_pair_count_;
    lowrank_ready_ = projected_pair_count_ >= 2;
    return true;
  }

  void freeze_pca_for_final_calibration_() {
    if (pca_frozen_) {
      return;
    }

    const bool frozen = set_projection_basis_from_online_pca_();
    (void) set_mean_direction_from_online_pca_();
    if (!frozen && projected_pair_count_ > 0) {
      projection_basis_ = eigvecs_.leftCols(opts_.J);
      projection_basis_ready_ = projection_basis_.allFinite();
    }

    projected_moments_.reset();
    online_pca_.reset();
    pca_frozen_ = true;
  }

  void finalize_transport_handoff_() {
    transport_endpoint_from_best_drop_ =
      transport_best_log_density_ - log_density_;
    if (std::isfinite(transport_endpoint_from_best_drop_) &&
        transport_endpoint_from_best_drop_ >
          opts_.transport_max_endpoint_from_best_drop) {
      rollback_transport_to_initial_state_();
      online_pca_.reset();
      return;
    }

    transport_handoff_pca_count_ = online_pca_.count();
    const bool ready = set_mean_direction_from_online_pca_(true);
    transport_handoff_pca_ready_ = ready && mean_direction_ready_;
    transport_handoff_pca_whitened_ =
      transport_handoff_pca_ready_ && mean_direction_whitened_;
    if (transport_handoff_pca_ready_) {
      transport_handoff_pca_basis_ = mean_direction_basis_;
      transport_handoff_pca_weights_ = mean_direction_weights_;
    } else {
      transport_handoff_pca_basis_.setZero();
      transport_handoff_pca_weights_.setZero();
    }
    online_pca_.reset();
  }

  void adapt_transport_warmup_(const std::size_t adaptation_draw) {
    if (adaptation_draw == 0 || adaptation_draw > opts_.warmup) {
      return;
    }
    (void) windowed_adaptation_.window_closed(adaptation_draw);
  }

  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    if (adaptation_draw == 0 || adaptation_draw > opts_.warmup) {
      return;
    }

    const bool calibrate_pca = pca_calibration_enabled_();
    if (calibrate_pca && !pca_frozen_ &&
        adaptation_draw >= pca_freeze_start_()) {
      freeze_pca_for_final_calibration_();
    }

    online_moments_.update(theta);
    if (calibrate_pca) {
      update_projected_moments_(theta);
      if (!pca_frozen_) {
        online_pca_.update(theta - mean_);
      }
    }

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();

      if (calibrate_pca && !pca_frozen_) {
        (void) activate_projected_pair_();
        const bool has_next_basis = set_projection_basis_from_online_pca_();
        (void) set_mean_direction_from_online_pca_();
        projected_moments_.reset();
        if (!has_next_basis) {
          projection_basis_ready_ = false;
        }
        online_pca_.reset();
      }
    }

    if (calibrate_pca && pca_frozen_ &&
        adaptation_draw == opts_.warmup) {
      activate_projected_pair_();
    }
  }

  std::pair<double, double> unpack_weibull_(const Eigen::VectorXd& eta) {
    const double shape = scale_from_log_(eta(0));
    const double scale = scale_from_log_(eta(1));
    return {shape, scale};
  }

  void set_bad_kl_(const Eigen::VectorXd& eta, double& value,
                   Eigen::VectorXd& grad) const {
    value = bad_kl_value_();
    grad = Eigen::VectorXd::Zero(eta.size());
    if (eta.size() > 1 && std::isfinite(eta(1))) {
      grad(1) = eta(1);
    }
  }

  static constexpr double bad_kl_value_() {
    return 1e100;
  }

  double relative_log_scale_(const double raw, const double log_s0) const {
    const double r = log_scale_radius_();
    if (!std::isfinite(raw)) {
      return log_s0;
    }
    return log_s0 + r * std::tanh(raw / r);
  }

  double relative_log_scale_derivative_(const double raw) const {
    const double r = log_scale_radius_();
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    const double th = std::tanh(raw / r);
    return 1.0 - th * th;
  }

  double scale_from_log_(double log_s) const {
    if (!std::isfinite(log_s)) {
      log_s = 0.0;
    }
    const double max_log = std::log(std::numeric_limits<double>::max()) - 2.0;
    const double min_log = std::log(std::numeric_limits<double>::min()) + 2.0;
    return std::exp(std::clamp(log_s, min_log, max_log)) + opts_.tol;
  }

  double bounded_log_shape_(const double raw) const {
    const double r = weibull_log_shape_radius_();
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    return r * std::tanh(raw / r);
  }

  double bounded_log_shape_derivative_(const double raw) const {
    const double r = weibull_log_shape_radius_();
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    const double th = std::tanh(raw / r);
    return 1.0 - th * th;
  }

  static constexpr double log_scale_radius_() {
    return 4.6051701859880918; // log(100)
  }

  static constexpr double weibull_log_shape_radius_() {
    return 2.3025850929940457; // log(10)
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
      return -std::numeric_limits<double>::infinity();
    }

    const int K = opts_.K;
    double log_density = -std::numeric_limits<double>::infinity();
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
      return -std::numeric_limits<double>::infinity();
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
    const double eps = std::numeric_limits<double>::epsilon();
    if (!std::isfinite(p)) {
      return 0.5;
    }
    return std::clamp(p, eps, 1.0 - eps);
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

  static constexpr double nan_() {
    return std::numeric_limits<double>::quiet_NaN();
  }

};

} // namespace klhr
