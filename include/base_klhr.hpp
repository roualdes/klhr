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

  struct LineFitDiagnostics {
    bool attempted = false;
    double mode_location = std::numeric_limits<double>::quiet_NaN();
    double final_location = std::numeric_limits<double>::quiet_NaN();
    double location_correction = std::numeric_limits<double>::quiet_NaN();
    double inverse_hessian = std::numeric_limits<double>::quiet_NaN();
    bool hessian_usable = false;
    bool hessian_identity = false;
    double laplace_scale = std::numeric_limits<double>::quiet_NaN();
    double initial_scale = std::numeric_limits<double>::quiet_NaN();
    double final_scale = std::numeric_limits<double>::quiet_NaN();
    double laplace_log_scale_correction =
      std::numeric_limits<double>::quiet_NaN();
    double laplace_scale_ratio = std::numeric_limits<double>::quiet_NaN();
    double log_scale_correction = std::numeric_limits<double>::quiet_NaN();
    double scale_ratio = std::numeric_limits<double>::quiet_NaN();
    double scale_bound_fraction = std::numeric_limits<double>::quiet_NaN();
    double scale_transform_derivative =
      std::numeric_limits<double>::quiet_NaN();
    bool scale_saturated = false;
    double final_skew = std::numeric_limits<double>::quiet_NaN();
    bool mode_success = false;
    std::size_t mode_iterations = 0;
    std::size_t mode_nfev = 0;
    bool kl_attempted = false;
    bool kl_success = false;
    std::size_t kl_iterations = 0;
    std::size_t kl_nfev = 0;
    double kl_initial_objective = std::numeric_limits<double>::quiet_NaN();
    double kl_final_objective = std::numeric_limits<double>::quiet_NaN();
    double kl_objective_improvement =
      std::numeric_limits<double>::quiet_NaN();
    double kl_initial_gradient_norm =
      std::numeric_limits<double>::quiet_NaN();
    double kl_final_gradient_norm =
      std::numeric_limits<double>::quiet_NaN();
  };

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
    windowed_adaptation_(opts_.warmup, opts_.windowsize,
                         opts_.windowscale),
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

    const Eigen::Index D = dim();
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J + 1);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J + 1);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_weights_ = Eigen::VectorXd::Ones(opts_.J);
    nfev_ = 0;
    acceptance_rate_ = 0.0;
    Eigen::VectorXd initial_grad = Eigen::VectorXd::Zero(D);
    bsm_.log_density_gradient_noe(theta_, log_density_, initial_grad);
    ++nfev_;
    transport_.initialize({theta_, initial_grad, log_density_,
                           theta_.allFinite() && initial_grad.allFinite() &&
                             std::isfinite(log_density_)},
                          rng_, std_normal_);
    draw_ = 0;
  }

  virtual ~BaseKLHR() = default;

  std::size_t dim() {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  Eigen::VectorXd draw() {
    current_forward_line_fit_ = LineFitDiagnostics{};
    current_reverse_line_fit_ = LineFitDiagnostics{};
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      auto result = transport_.step(
        bsm_, rng_, std_uniform_, std_normal_,
        [this](const Eigen::VectorXd& theta) {
          return constrain_or_missing_(theta);
        });
      nfev_ += result.evaluations;
      theta_ = result.state.theta;
      log_density_ = result.state.log_density;
      const double acceptance_delta =
        static_cast<double>(result.moved) - acceptance_rate_;
      acceptance_rate_ += acceptance_delta / draw_;
      const Eigen::VectorXd proposal = result.proposal.size() > 0 ?
        result.proposal : missing_constrained_draw_();
      record_proposal_step_(proposal, result.proposal_log_density, nan_(),
                            result.moved, result.moved);
      record_kl_step_(Eigen::VectorXd::Constant(3, nan_()), nan_(), false);
      adapt_transport_warmup_(draw_);
      if (draw_ == opts_.initial_transport_steps) {
        apply_transport_handoff_(transport_.finish(rng_, std_normal_));
      }
      record_line_fit_diagnostics_();
      return bsm_.param_constrain(theta_);
    }

    Eigen::VectorXd rho = random_direction();
    regular_kl_step_(rho);
    transport_.record_inactive_step();
    adapt_warmup_(theta_, draw_);
    record_line_fit_diagnostics_();
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

  const std::vector<LineFitDiagnostics>&
  forward_line_fit_diagnostics_history() const {
    return forward_line_fit_diagnostics_;
  }

  const std::vector<LineFitDiagnostics>&
  reverse_line_fit_diagnostics_history() const {
    return reverse_line_fit_diagnostics_;
  }

  const std::vector<double>& transport_distance_history() const {
    return transport_.distance_history();
  }

  const std::vector<double>& transport_reflections_history() const {
    return transport_.reflections_history();
  }

  const std::vector<double>& transport_logp_gain_history() const {
    return transport_.logp_gain_history();
  }

  const std::vector<double>& transport_uturn_history() const {
    return transport_.uturn_history();
  }

  const std::vector<double>& transport_moved_history() const {
    return transport_.moved_history();
  }

  const std::vector<Eigen::VectorXd>& transport_variance_history() const {
    return transport_.variance_history();
  }

  const std::vector<double>& transport_direction_norm_history() const {
    return transport_.direction_norm_history();
  }

  const Eigen::MatrixXd& transport_handoff_pca_basis() const {
    return transport_.handoff_pca_basis();
  }

  const Eigen::VectorXd& transport_handoff_pca_weights() const {
    return transport_.handoff_pca_weights();
  }

  std::size_t transport_handoff_pca_count() const {
    return transport_.handoff_pca_count();
  }

  bool transport_handoff_pca_ready() const {
    return transport_.handoff_pca_ready();
  }

  bool transport_handoff_pca_whitened() const {
    return transport_.handoff_pca_whitened();
  }

  bool transport_rollback() const {
    return transport_.rollback();
  }

  double transport_initial_log_density() const {
    return transport_.initial_log_density();
  }

  double transport_best_log_density() const {
    return transport_.best_log_density();
  }

  double transport_endpoint_from_best_drop() const {
    return transport_.endpoint_from_best_drop();
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

  struct LineFitResult {
    Eigen::VectorXd eta;
    LineFitDiagnostics diagnostics;
  };

  virtual LineFitResult fit_line_(const Eigen::VectorXd& center,
                                  const Eigen::VectorXd& rho) = 0;

  virtual double overrelaxed_proposal_(const Eigen::VectorXd& eta) = 0;

  virtual double transition_density_(const double from, const double to,
                                     const Eigen::VectorXd& eta) = 0;

  virtual void record_kl_step_(const Eigen::VectorXd& eta, const double xi,
                               const bool accepted) {
    (void) eta;
    (void) xi;
    (void) accepted;
  }

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
    double inverse_hessian = std::numeric_limits<double>::quiet_NaN();
    bool hessian_usable = false;
    bool hessian_identity = false;
    bool success = false;
    std::size_t iterations = 0;
    std::size_t nfev = 0;
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
    if (mode.hess_inv.rows() == 1 && mode.hess_inv.cols() == 1) {
      out.inverse_hessian = mode.hess_inv(0, 0);
    }
    out.hessian_usable =
      std::isfinite(out.inverse_hessian) && out.inverse_hessian > 0.0;
    if (out.hessian_usable) {
      out.log_scale = 0.5 * std::log(out.inverse_hessian);
      const double identity_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(out.inverse_hessian));
      out.hessian_identity =
        std::abs(out.inverse_hessian - 1.0) <= identity_tolerance;
    }
    out.success = mode.success;
    out.iterations = mode.nit;
    out.nfev = mode.nfev;
    return out;
  }

  LineFitResult make_line_fit_result_(
      const LineModeEstimate& mode,
      const bfgs::BfgsResult& fit,
      const Eigen::VectorXd& raw,
      const Eigen::VectorXd& eta,
      const double initial_objective,
      const double initial_gradient_norm) const {
    LineFitResult out = make_line_fit_result_(mode, raw, eta);
    LineFitDiagnostics& diagnostics = out.diagnostics;
    diagnostics.kl_attempted = true;
    diagnostics.kl_success = fit.success;
    diagnostics.kl_iterations = fit.nit;
    diagnostics.kl_nfev =
      fit.nfev * static_cast<std::size_t>(opts_.N);
    diagnostics.kl_initial_objective = initial_objective;
    diagnostics.kl_initial_gradient_norm = initial_gradient_norm;

    const bool fit_result_matches_output =
      fit.x.size() == raw.size() && fit.x.allFinite();
    diagnostics.kl_final_objective = fit_result_matches_output ?
      fit.fun : initial_objective;
    if (std::isfinite(diagnostics.kl_initial_objective) &&
        std::isfinite(diagnostics.kl_final_objective)) {
      diagnostics.kl_objective_improvement =
        diagnostics.kl_initial_objective - diagnostics.kl_final_objective;
    }
    if (fit.jac.size() == raw.size() && fit.jac.allFinite()) {
      diagnostics.kl_final_gradient_norm =
        fit.jac.lpNorm<Eigen::Infinity>();
    }
    return out;
  }

  LineFitResult make_laplace_line_fit_result_(
      const LineModeEstimate& mode,
      const Eigen::Index parameter_count) const {
    Eigen::VectorXd raw = Eigen::VectorXd::Zero(parameter_count);
    Eigen::VectorXd eta = Eigen::VectorXd::Zero(parameter_count);
    raw(0) = mode.mode;
    eta(0) = mode.mode;
    eta(1) = mode.log_scale;
    return make_line_fit_result_(mode, raw, eta);
  }

  LineFitResult make_line_fit_result_(
      const LineModeEstimate& mode,
      const Eigen::VectorXd& raw,
      const Eigen::VectorXd& eta) const {
    LineFitResult out;
    out.eta = eta;

    LineFitDiagnostics& diagnostics = out.diagnostics;
    diagnostics.attempted = true;
    diagnostics.mode_location = mode.mode;
    diagnostics.inverse_hessian = mode.inverse_hessian;
    diagnostics.hessian_usable = mode.hessian_usable;
    diagnostics.hessian_identity = mode.hessian_identity;
    diagnostics.mode_success = mode.success;
    diagnostics.mode_iterations = mode.iterations;
    diagnostics.mode_nfev = mode.nfev;
    if (mode.hessian_usable) {
      diagnostics.laplace_scale = std::sqrt(mode.inverse_hessian);
    }
    diagnostics.initial_scale = scale_from_log_(mode.log_scale);

    if (eta.size() >= 2 && eta.allFinite()) {
      diagnostics.final_location = eta(0);
      diagnostics.location_correction = eta(0) - mode.mode;
      diagnostics.final_scale = scale_from_log_(eta(1));
      if (mode.hessian_usable) {
        const double laplace_log_scale =
          0.5 * std::log(mode.inverse_hessian);
        diagnostics.laplace_log_scale_correction =
          eta(1) - laplace_log_scale;
        diagnostics.laplace_scale_ratio =
          std::exp(diagnostics.laplace_log_scale_correction);
      }
      diagnostics.log_scale_correction = eta(1) - mode.log_scale;
      diagnostics.scale_ratio = std::exp(diagnostics.log_scale_correction);
      diagnostics.scale_bound_fraction =
        std::abs(diagnostics.log_scale_correction) / log_scale_radius_();
      diagnostics.scale_saturated =
        diagnostics.scale_bound_fraction >= scale_saturation_fraction_();
      if (eta.size() >= 3) {
        diagnostics.final_skew = eta(2);
      }
    }
    if (raw.size() >= 2 && std::isfinite(raw(1))) {
      diagnostics.scale_transform_derivative =
        relative_log_scale_derivative_(raw(1));
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
  std::vector<Eigen::VectorXd> proposal_draws_;
  std::vector<double> proposal_log_accept_;
  std::vector<double> proposal_log_density_;
  std::vector<double> proposal_accepted_;
  std::vector<double> proposal_valid_;
  std::vector<LineFitDiagnostics> forward_line_fit_diagnostics_;
  std::vector<LineFitDiagnostics> reverse_line_fit_diagnostics_;
  LineFitDiagnostics current_forward_line_fit_;
  LineFitDiagnostics current_reverse_line_fit_;
  bool projection_basis_ready_ = false;
  bool mean_direction_ready_ = false;
  bool mean_direction_whitened_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  std::size_t projected_pair_count_ = 0;

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

    LineFitResult forward_fit = fit_line_(theta_, rho);
    current_forward_line_fit_ = forward_fit.diagnostics;
    Eigen::VectorXd eta = std::move(forward_fit.eta);
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

    LineFitResult reverse_fit = fit_line_(thetap, rho);
    current_reverse_line_fit_ = reverse_fit.diagnostics;
    Eigen::VectorXd reta = std::move(reverse_fit.eta);
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

  Eigen::VectorXd missing_constrained_draw_() {
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

  void record_line_fit_diagnostics_() {
    forward_line_fit_diagnostics_.push_back(current_forward_line_fit_);
    reverse_line_fit_diagnostics_.push_back(current_reverse_line_fit_);
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

    online_moments_.reset();
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

  void apply_transport_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    theta_ = handoff.state.theta;
    log_density_ = handoff.state.log_density;
    if (handoff.rollback) {
      reset_adaptation_to_defaults_(true);
      online_pca_.reset();
      return;
    }

    if (handoff.covariance.size() == static_cast<Eigen::Index>(dim()) &&
        handoff.covariance.allFinite()) {
      cov_ = handoff.covariance;
    }
    mean_direction_ready_ = handoff.pca_ready;
    mean_direction_whitened_ = handoff.pca_whitened;
    if (handoff.pca_ready &&
        handoff.pca_basis.rows() == static_cast<Eigen::Index>(dim()) &&
        handoff.pca_basis.cols() == opts_.J &&
        handoff.pca_weights.size() == opts_.J &&
        handoff.pca_basis.allFinite() &&
        handoff.pca_weights.allFinite()) {
      mean_direction_basis_ = handoff.pca_basis;
      mean_direction_weights_ = handoff.pca_weights;
    } else {
      mean_direction_basis_.setZero();
      mean_direction_weights_.setOnes();
      mean_direction_ready_ = false;
      mean_direction_whitened_ = false;
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

  void set_bad_kl_(const Eigen::VectorXd& eta, double& value,
                   Eigen::VectorXd& grad) const {
    numerics::set_bad_kl(eta, value, grad);
  }

  static constexpr double bad_kl_value_() {
    return numerics::bad_kl_value();
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

  static constexpr double log_scale_radius_() {
    return numerics::log_scale_radius();
  }

  static constexpr double scale_saturation_fraction_() {
    return 0.9;
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

  static constexpr double nan_() {
    return std::numeric_limits<double>::quiet_NaN();
  }

};

} // namespace klhr
