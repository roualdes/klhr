#pragma once

#include "bridgestan.hpp"
#include "gausshermite.hpp"
#include "onlinepca.hpp"
#include "rng.hpp"
#include "welford.hpp"
#include "windowedadaptation.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace klhr {

enum class TransportProposal {
  Overrelaxed,
  Random
};

enum class TransportDirectionLaw {
  Kappa,
  Projected
};

struct KlhrOptions {
  double stepsize = 1.0;
  std::uint64_t seed = 0;
  std::size_t N = 8;
  double tol = 1e-10;
  double grad_clip = std::numeric_limits<double>::infinity(); // 1e15; //
  double scale_clip = 300;
  double gtol = 1e-3;
  std::size_t K = 16;
  double initscale = 0.1;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t J = 2;
  double l = 0.0;
  std::size_t initial_fast_adaptation_steps = 100; // 100
  std::size_t initial_transport_gradient_history = 3;
  double initial_transport_gradient_projection_probability = 0.5;
  double initial_transport_gradient_floor = 1e-8;
  double initial_transport_direction_kappa = 10.0;
  TransportDirectionLaw initial_transport_direction_law =
    TransportDirectionLaw::Kappa;
  TransportProposal initial_transport_proposal =
    TransportProposal::Random;
};

class BaseKLHR {
public:

  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;
  std::size_t stop_transport_idx_;
  int diagnostic_phase_;
  double diagnostic_grad_dot_move_;
  double diagnostic_cos_grad_move_;
  double diagnostic_beta_slope_;
  double diagnostic_logp_gain_;
  double diagnostic_diag_jump_;
  double diagnostic_move_norm_;
  double diagnostic_grad_norm_;
  std::size_t diagnostic_transport_direction_attempts_;
  Eigen::VectorXd diagnostic_gradient_;
  Eigen::VectorXd diagnostic_move_;

  BaseKLHR(std::string stan_file, std::string json_file,
           const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    uniform_uint_(),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(options),
    windowed_adaptation_(options.warmup - std::min(options.initial_fast_adaptation_steps,
                                                   options.warmup),
                         options.windowsize,
                         options.windowscale),
    online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), options.J, options.l, options.tol) {

    if (opts_.seed == 0) {
      std::random_device rd;
      rng_.seed(rd());
    }

    bsrng_ = bsm_.make_rng(uniform_uint_(rng_));
    theta_.resize(dim());
    theta_ = bsm_.param_initialize(bsrng_);

    w_.resize(opts_.N);
    x_.resize(opts_.N);
    gauss_hermite(opts_.N, w_, x_);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);

    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    diagnostic_gradient_ = Eigen::VectorXd::Zero(D);
    diagnostic_move_ = Eigen::VectorXd::Zero(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, static_cast<Eigen::Index>(opts_.J + 1));
    eigvals_ = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(opts_.J + 1));
    transport_gradients_.reserve(opts_.initial_transport_gradient_history);

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    log_density_ = bsm_.log_density_noe(theta_);
    ++nfev_;
    stop_transport_idx_ = 0;
    draw_ = 0;
    mh_draw_ = 0;
    warmup_draw_ = 0;
    reset_diagnostics_();
    initial_fast_adaptation_steps_ =
      std::min(opts_.initial_fast_adaptation_steps, opts_.warmup);
    adaptation_warmup_ = opts_.warmup - initial_fast_adaptation_steps_;
  }

  virtual ~BaseKLHR() = default;

  std::size_t dim() {
    return bsm_.dim();
  }

  static double log_sum_exp_2_(double a, double b) {
    if (!std::isfinite(a)) {
      return b;
    }
    if (!std::isfinite(b)) {
      return a;
    }
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
  }

  Eigen::VectorXd KL_step() {
    return regular_kl_step_();
  }

  Eigen::VectorXd KL_step(const Eigen::VectorXd& rho) {
    return regular_kl_step_(rho, nullptr);
  }

  Eigen::VectorXd Metropolis_step() {
    Eigen::VectorXd thetap = theta_ + normal_(dim()) * opts_.stepsize;
    double proposal_logp;
    bsm_.log_density_noe(thetap, proposal_logp);
    ++nfev_;
    double r = proposal_logp - log_density_;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    if (a) {
      theta_ = thetap;
      log_density_ = proposal_logp;
    }
    return theta_;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    Eigen::VectorXd theta;
    if (draw_ <= initial_fast_adaptation_steps_) {
      theta = initial_transport_step_();
      if (draw_ == initial_fast_adaptation_steps_) {
        stop_transport_idx_ = draw_;
      }
    } else {
      ++warmup_draw_;
      theta = regular_kl_step_();
      adapt_warmup_(theta, warmup_draw_);
    }
    return bsm_.param_constrain(theta);
  }

  Eigen::VectorXd random_direction() {
    Eigen::VectorXd weights = eigvals_.cwiseMax(0.0);
    std::discrete_distribution<std::size_t> component(weights.data(),
                                                      weights.data() + weights.size());
    const std::size_t j = component(rng_);

    Eigen::VectorXd rho(dim());
    const Eigen::VectorXd center = eigvecs_.col(static_cast<Eigen::Index>(j));
    for (Eigen::Index d = 0; d < rho.size(); ++d) {
      const double variance = std::max(cov_(d), opts_.tol);
      rho(d) = center(d) + std::sqrt(variance) * std_normal_(rng_);
    }

    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_(dim());
      norm = rho.norm();
    }
    return rho / (norm + opts_.tol);
  }

protected:

  struct StepStats {
    double accept_prob = 0.0;
    double scaled_jump = 0.0;
    bool accepted = false;
  };

  struct TransportStepProposal {
    Eigen::VectorXd theta;
    Eigen::VectorXd gradient;
    double logp = -std::numeric_limits<double>::infinity();
    double log_accept_ratio = -std::numeric_limits<double>::infinity();
    double logp_gain = std::numeric_limits<double>::quiet_NaN();
    double diag_jump = std::numeric_limits<double>::quiet_NaN();
  };

  virtual Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                                    const Eigen::VectorXd& rho) = 0;

  virtual double overrelaxed_line_proposal_(const Eigen::VectorXd& eta) = 0;

  virtual double transport_line_proposal_(const Eigen::VectorXd& eta) = 0;

  virtual double log_line_transition_density_(double from,
                                              double to,
                                              const Eigen::VectorXd& eta) = 0;

  virtual double log_transport_radial_density_(
      double r,
      const Eigen::VectorXd& eta) = 0;

  virtual double reference_scale_(const Eigen::VectorXd& eta) = 0;

  mcmcpp::bsmodel bsm_;
  mcmcpp::bsrng bsrng_;
  mcmcpp::rng rng_;

  std::uniform_int_distribution<unsigned int> uniform_uint_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  WindowedAdaptation windowed_adaptation_;
  WelfordAccumulator online_moments_;
  OnlinePCA online_pca_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Guass-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;

  std::size_t draw_;
  std::size_t mh_draw_;
  std::size_t warmup_draw_;
  std::size_t initial_fast_adaptation_steps_;
  std::size_t adaptation_warmup_;
  std::vector<Eigen::VectorXd> transport_gradients_;
  std::size_t transport_gradient_next_ = 0;

  Eigen::VectorXd regular_kl_step_() {
    return regular_kl_step_(random_direction(), nullptr);
  }

  Eigen::VectorXd regular_kl_step_(const Eigen::VectorXd& rho,
                                   StepStats* stats) {
    const Eigen::VectorXd theta_before = theta_;
    Eigen::VectorXd grad(dim());
    double current_logp = log_density_;
    bsm_.log_density_gradient_noe(theta_, current_logp, grad);
    ++nfev_;
    if (std::isfinite(current_logp)) {
      log_density_ = current_logp;
    } else {
      current_logp = log_density_;
    }
    sanitize_gradient_(grad);

    Eigen::VectorXd eta = fit_line_(theta_before, rho);
    const double xi = overrelaxed_line_proposal_(eta);

    Eigen::VectorXd thetap = theta_;
    double proposal_logp = log_density_;
    double r = 0.0;
    if (!std::isfinite(xi)) {
      proposal_logp = -std::numeric_limits<double>::infinity();
      r = -std::numeric_limits<double>::infinity();
    } else if (std::abs(xi) > opts_.tol) {
      thetap = xi * rho + theta_;
      proposal_logp = -std::numeric_limits<double>::infinity();
      r = -std::numeric_limits<double>::infinity();
    }
    if (std::isfinite(xi) && std::abs(xi) > opts_.tol &&
        thetap.allFinite()) {
      proposal_logp = bsm_.log_density_noe(thetap);
      ++nfev_;
      if (std::isfinite(proposal_logp)) {
        const double forward_transition =
          log_line_transition_density_(0.0, xi, eta);
        Eigen::VectorXd reverse_eta = fit_line_(thetap, rho);
        const double reverse_transition =
          log_line_transition_density_(0.0, -xi, reverse_eta);
        if (std::isfinite(forward_transition) &&
            std::isfinite(reverse_transition)) {
          r = proposal_logp - log_density_;
          r += reverse_transition;
          r -= forward_transition;
        }
      }
    }

    const double accept_prob = std::exp(std::min(0.0, r));
    const bool accepted = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    ++mh_draw_;
    const double d = static_cast<double>(accepted) - acceptance_rate_;
    acceptance_rate_ += d / mh_draw_;
    if (accepted) {
      theta_ = thetap;
      log_density_ = proposal_logp;
    }
    if (stats != nullptr) {
      stats->accept_prob = accept_prob;
      stats->accepted = accepted;
      stats->scaled_jump = std::abs(xi) / reference_scale_(eta);
    }
    const int phase = warmup_draw_ <= adaptation_warmup_ ? 1 : 2;
    record_move_diagnostics_(phase, grad, theta_before, theta_,
                             current_logp, log_density_);
    return theta_;
  }

  TransportStepProposal make_transport_klhr_proposal_(
      const Eigen::VectorXd& rho_in,
      const Eigen::VectorXd& theta_before,
      double current_logp,
      const Eigen::VectorXd& current_grad) {
    Eigen::VectorXd rho = normalized_direction_(rho_in);
    Eigen::VectorXd eta = fit_line_(theta_before, rho);
    double xi = transport_line_proposal_(eta);
    if (!std::isfinite(xi)) {
      return {};
    }
    if (xi < 0.0) {
      xi = -xi;
      rho = -rho;
    }
    const Eigen::VectorXd theta_proposed = theta_before + xi * rho;
    if (!theta_proposed.allFinite()) {
      return {};
    }

    Eigen::VectorXd proposal_grad(dim());
    double proposal_logp = -std::numeric_limits<double>::infinity();
    bsm_.log_density_gradient_noe(theta_proposed, proposal_logp,
                                  proposal_grad);
    ++nfev_;
    if (!std::isfinite(proposal_logp)) {
      return {};
    }
    sanitize_gradient_(proposal_grad);

    const Eigen::VectorXd reverse_rho = -rho;
    Eigen::VectorXd reverse_eta = fit_line_(theta_proposed, reverse_rho);

    const double forward_line = log_transport_radial_density_(xi, eta);
    const double reverse_line =
      log_transport_radial_density_(xi, reverse_eta);
    if (!std::isfinite(forward_line) || !std::isfinite(reverse_line)) {
      return {};
    }

    double r = proposal_logp - current_logp;
    r += transport_direction_log_density_(reverse_rho, proposal_grad);
    r -= transport_direction_log_density_(rho, current_grad);
    r += reverse_line;
    r -= forward_line;
    const double jump = diag_scaled_distance_(theta_proposed - theta_before,
                                              theta_before);

    TransportStepProposal proposal;
    proposal.theta = theta_proposed;
    proposal.gradient = proposal_grad;
    proposal.logp = proposal_logp;
    proposal.log_accept_ratio = r;
    proposal.logp_gain = proposal_logp - current_logp;
    proposal.diag_jump = jump;
    return proposal;
  }

  Eigen::VectorXd initial_transport_step_() {
    const Eigen::VectorXd theta_before = theta_;
    Eigen::VectorXd grad(dim());
    double current_logp = log_density_;
    bsm_.log_density_gradient_noe(theta_, current_logp, grad);
    ++nfev_;
    if (std::isfinite(current_logp)) {
      log_density_ = current_logp;
    } else {
      current_logp = log_density_;
    }
    sanitize_gradient_(grad);

    const Eigen::VectorXd rho = tangent_transport_direction_(grad);
    TransportStepProposal proposal =
      make_transport_klhr_proposal_(rho, theta_before, current_logp, grad);
    const double log_accept = std::isfinite(proposal.log_accept_ratio) ?
      std::min(0.0, proposal.log_accept_ratio) :
      -std::numeric_limits<double>::infinity();
    const bool accepted = std::log(std_uniform_(rng_)) <
      log_accept;
    ++mh_draw_;
    const double d = static_cast<double>(accepted) - acceptance_rate_;
    acceptance_rate_ += d / mh_draw_;
    if (accepted && proposal.theta.allFinite() && std::isfinite(proposal.logp)) {
      theta_ = proposal.theta;
      log_density_ = proposal.logp;
      if (opts_.initial_transport_direction_law ==
          TransportDirectionLaw::Projected) {
        update_transport_gradient_history_(proposal.gradient);
      }
    }
    record_move_diagnostics_(0, grad, theta_before, theta_, current_logp,
                             log_density_, accepted ? proposal.diag_jump : 0.0);
    return theta_;
  }

  void sanitize_gradient_(Eigen::VectorXd& grad) const {
    for (Eigen::Index d = 0; d < grad.size(); ++d) {
      if (!std::isfinite(grad(d))) {
        grad(d) = 0.0;
      } else {
        grad(d) = std::clamp(grad(d), -opts_.grad_clip, opts_.grad_clip);
      }
    }
  }

  Eigen::VectorXd tangent_transport_direction_(const Eigen::VectorXd& grad) {
    if (opts_.initial_transport_direction_law ==
        TransportDirectionLaw::Projected) {
      diagnostic_transport_direction_attempts_ = 1;
      return projected_transport_direction_(grad);
    }

    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    const double kappa = std::max(0.0, opts_.initial_transport_direction_kappa);
    if (kappa <= 0.0 || !has_usable_gradient_(grad)) {
      diagnostic_transport_direction_attempts_ = 1;
      return normalized_direction_(normal_(D));
    }

    for (std::size_t attempt = 0; attempt < 10'000; ++attempt) {
      Eigen::VectorXd rho = normalized_direction_(normal_(D));
      const double log_accept = transport_direction_log_density_(rho, grad);
      if (std::log(std_uniform_(rng_)) <= log_accept) {
        diagnostic_transport_direction_attempts_ = attempt + 1;
        return rho;
      }
    }

    diagnostic_transport_direction_attempts_ = 10'000;
    Eigen::VectorXd rho = normalized_direction_(normal_(D));
    const Eigen::VectorXd normal = grad / grad.norm();
    rho.noalias() -= normal.dot(rho) * normal;
    const double norm = rho.norm();
    if (std::isfinite(norm) && norm > opts_.tol) {
      return rho / norm;
    }
    return normalized_direction_(normal_(D));
  }

  Eigen::VectorXd projected_transport_direction_(const Eigen::VectorXd& grad) {
    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    for (std::size_t attempt = 0; attempt < 4; ++attempt) {
      Eigen::VectorXd rho = normal_(D);
      rho = project_from_transport_avoidance_span_(rho, grad);
      const double norm = rho.norm();
      if (std::isfinite(norm) && norm > opts_.tol) {
        return rho / norm;
      }
    }

    for (Eigen::Index d = 0; d < D; ++d) {
      Eigen::VectorXd rho = Eigen::VectorXd::Unit(D, d);
      rho = project_from_transport_avoidance_span_(rho, grad);
      const double norm = rho.norm();
      if (std::isfinite(norm) && norm > opts_.tol) {
        return rho / norm;
      }
    }

    return normalized_direction_(normal_(D));
  }

  Eigen::VectorXd project_from_transport_avoidance_span_(
      const Eigen::VectorXd& direction,
      const Eigen::VectorXd& grad) {
    Eigen::VectorXd tangent = direction;
    if (grad.size() != tangent.size()) {
      return tangent;
    }
    const double floor = std::max(opts_.initial_transport_gradient_floor,
                                  opts_.tol);
    std::vector<Eigen::VectorXd> basis;
    basis.reserve(std::min<std::size_t>(
      transport_gradients_.size() + 1,
      tangent.size() > 0 ? static_cast<std::size_t>(tangent.size()) : 0));

    auto project_normal = [&](const Eigen::VectorXd& normal_in) {
      if (normal_in.size() != tangent.size()) {
        return;
      }
      if (basis.size() + 1 >= static_cast<std::size_t>(tangent.size())) {
        return;
      }
      Eigen::VectorXd normal = normal_in;
      for (const Eigen::VectorXd& b : basis) {
        normal.noalias() -= b.dot(normal) * b;
      }
      const double norm = normal.norm();
      if (!std::isfinite(norm) || norm <= floor) {
        return;
      }
      normal /= norm;
      tangent.noalias() -= normal.dot(tangent) * normal;
      basis.push_back(normal);
    };

    project_normal(grad);
    double projection_probability =
      opts_.initial_transport_gradient_projection_probability;
    if (!std::isfinite(projection_probability)) {
      projection_probability = 1.0;
    }
    projection_probability = std::clamp(projection_probability, 0.0, 1.0);
    for (const Eigen::VectorXd& recent_grad : transport_gradients_) {
      if (projection_probability >= 1.0 ||
          std_uniform_(rng_) < projection_probability) {
        project_normal(recent_grad);
      }
    }

    return tangent;
  }

  bool has_usable_gradient_(const Eigen::VectorXd& grad) const {
    const double floor = std::max(opts_.initial_transport_gradient_floor,
                                  opts_.tol);
    const double norm = grad.norm();
    return grad.size() > 0 && std::isfinite(norm) && norm > floor;
  }

  double transport_direction_log_density_(const Eigen::VectorXd& rho_in,
                                          const Eigen::VectorXd& grad) const {
    if (opts_.initial_transport_direction_law ==
        TransportDirectionLaw::Projected) {
      return 0.0;
    }
    const double kappa = std::max(0.0, opts_.initial_transport_direction_kappa);
    if (kappa <= 0.0 || !has_usable_gradient_(grad)) {
      return 0.0;
    }
    const Eigen::VectorXd rho = normalized_direction_const_(rho_in);
    const Eigen::VectorXd normal = grad / grad.norm();
    const double cosine = normal.dot(rho);
    return -kappa * cosine * cosine;
  }

  void update_transport_gradient_history_() {
    if (opts_.initial_transport_gradient_history == 0) {
      return;
    }

    Eigen::VectorXd new_grad(dim());
    double new_logp = log_density_;
    bsm_.log_density_gradient_noe(theta_, new_logp, new_grad);
    ++nfev_;
    if (!std::isfinite(new_logp)) {
      return;
    }
    log_density_ = new_logp;
    update_transport_gradient_history_(new_grad);
  }

  void update_transport_gradient_history_(const Eigen::VectorXd& grad) {
    if (opts_.initial_transport_gradient_history == 0) {
      return;
    }
    if (grad.size() != static_cast<Eigen::Index>(dim())) {
      return;
    }
    Eigen::VectorXd new_grad = grad;
    sanitize_gradient_(new_grad);
    const double norm = new_grad.norm();
    const double floor = std::max(opts_.initial_transport_gradient_floor,
                                  opts_.tol);
    if (!std::isfinite(norm) || norm <= floor) {
      return;
    }

    if (transport_gradients_.size() <
        opts_.initial_transport_gradient_history) {
      transport_gradients_.push_back(new_grad);
      return;
    }

    transport_gradients_[transport_gradient_next_] = new_grad;
    transport_gradient_next_ =
      (transport_gradient_next_ + 1) %
      opts_.initial_transport_gradient_history;
  }

  double diag_scaled_distance_(const Eigen::VectorXd& delta,
                               const Eigen::VectorXd& center) const {
    if (delta.size() == 0 || delta.size() != center.size()) {
      return 0.0;
    }

    double dist_sq = 0.0;
    for (Eigen::Index d = 0; d < delta.size(); ++d) {
      const double scale = std::max(std::abs(center(d)), 1.0);
      const double z = delta(d) / scale;
      dist_sq += z * z;
    }
    if (!std::isfinite(dist_sq) || dist_sq <= 0.0) {
      return 0.0;
    }
    return std::sqrt(dist_sq);
  }

  void reset_diagnostics_() {
    diagnostic_phase_ = -1;
    diagnostic_grad_dot_move_ = quiet_nan_();
    diagnostic_cos_grad_move_ = quiet_nan_();
    diagnostic_beta_slope_ = quiet_nan_();
    diagnostic_logp_gain_ = quiet_nan_();
    diagnostic_diag_jump_ = quiet_nan_();
    diagnostic_move_norm_ = quiet_nan_();
    diagnostic_grad_norm_ = quiet_nan_();
    diagnostic_transport_direction_attempts_ = 0;
    diagnostic_gradient_.setZero();
    diagnostic_move_.setZero();
  }

  void record_move_diagnostics_(int phase,
                                const Eigen::VectorXd& grad,
                                const Eigen::VectorXd& theta_before,
                                const Eigen::VectorXd& theta_after,
                                double logp_before,
                                double logp_after,
                                double diag_jump = quiet_nan_()) {
    diagnostic_phase_ = phase;
    diagnostic_gradient_ = grad;
    diagnostic_move_ = theta_after - theta_before;
    diagnostic_grad_norm_ = grad.norm();
    diagnostic_move_norm_ = diagnostic_move_.norm();
    diagnostic_grad_dot_move_ = grad.dot(diagnostic_move_);
    diagnostic_logp_gain_ = logp_after - logp_before;
    diagnostic_diag_jump_ = std::isfinite(diag_jump) ?
      diag_jump : diag_scaled_distance_(diagnostic_move_, theta_before);

    if (diagnostic_grad_norm_ > opts_.tol &&
        diagnostic_move_norm_ > opts_.tol) {
      diagnostic_cos_grad_move_ =
        diagnostic_grad_dot_move_ / (diagnostic_grad_norm_ * diagnostic_move_norm_);
    } else {
      diagnostic_cos_grad_move_ = quiet_nan_();
    }

    if (diagnostic_move_.size() >= 2 &&
        std::abs(diagnostic_move_(1)) > opts_.tol) {
      diagnostic_beta_slope_ = diagnostic_move_(0) / diagnostic_move_(1);
    } else {
      diagnostic_beta_slope_ = quiet_nan_();
    }

    if (phase != 0) {
      diagnostic_transport_direction_attempts_ = 0;
    }
  }

  static constexpr double quiet_nan_() {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Eigen::VectorXd normalized_direction_(const Eigen::VectorXd& direction) {
    Eigen::VectorXd rho = direction;
    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_(static_cast<Eigen::Index>(dim()));
      norm = rho.norm();
    }
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = Eigen::VectorXd::Unit(static_cast<Eigen::Index>(dim()), 0);
      norm = 1.0;
    }
    return rho / norm;
  }

  Eigen::VectorXd normalized_direction_const_(
      const Eigen::VectorXd& direction) const {
    Eigen::VectorXd rho = direction;
    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = Eigen::VectorXd::Zero(direction.size());
      if (rho.size() > 0) {
        rho(0) = 1.0;
      }
      return rho;
    }
    return rho / norm;
  }

  void adapt_warmup_(const Eigen::VectorXd& theta, std::size_t warmup_draw) {
    if (warmup_draw > adaptation_warmup_) {
      return;
    }

    if (windowed_adaptation_.window_closed(warmup_draw)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();

      if (opts_.J > 0) {
        eigvecs_.leftCols(static_cast<Eigen::Index>(opts_.J)) = online_pca_.vectors();
        eigvals_.head(static_cast<Eigen::Index>(opts_.J)) = online_pca_.values();
      }
      online_pca_.reset();
    } else {
      online_moments_.update(theta);
      online_pca_.update(theta - mean_);
    }
  }

  double overrelaxed_cdf_proposal_(double u) {
    u = clamp_probability_(u);
    if (opts_.K == 0) {
      return clamp_probability_(std_uniform_(rng_));
    }

    const int K = static_cast<int>(std::min<std::size_t>(
      opts_.K, static_cast<std::size_t>(std::numeric_limits<int>::max())));

    std::binomial_distribution<int> binomial(K, u);
    const int r = binomial(rng_);

    double up = u;
    if (r > K - r) {
      const double v = beta_(static_cast<double>(K - r + 1),
                             static_cast<double>(2 * r - K));
      up = u * v;
    } else if (r < K - r) {
      const double v = beta_(static_cast<double>(r + 1),
                             static_cast<double>(K - 2 * r));
      up = 1.0 - (1.0 - u) * v;
    }

    return clamp_probability_(up);
  }

  double ordered_overrelaxed_cdf_log_density_(double from, double to) const {
    from = clamp_probability_(from);
    to = clamp_probability_(to);
    if (opts_.K == 0) {
      return 0.0;
    }
    if (from == to) {
      return -std::numeric_limits<double>::infinity();
    }

    const int K = static_cast<int>(std::min<std::size_t>(
      opts_.K, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    double log_density = -std::numeric_limits<double>::infinity();
    if (to < from) {
      const double log_from = std::log(from);
      const double v = to / from;
      for (int r = K / 2 + 1; r <= K; ++r) {
        const double a = static_cast<double>(K - r + 1);
        const double b = static_cast<double>(2 * r - K);
        const double term =
          log_binomial_pmf_(K, r, from) + log_beta_pdf_(v, a, b) -
          log_from;
        log_density = log_sum_exp_2_(log_density, term);
      }
    } else {
      const double log_one_minus_from = std::log1p(-from);
      const double v = (1.0 - to) / (1.0 - from);
      for (int r = 0; r < (K + 1) / 2; ++r) {
        const double a = static_cast<double>(r + 1);
        const double b = static_cast<double>(K - 2 * r);
        const double term =
          log_binomial_pmf_(K, r, from) + log_beta_pdf_(v, a, b) -
          log_one_minus_from;
        log_density = log_sum_exp_2_(log_density, term);
      }
    }
    return log_density;
  }

  static double log_binomial_pmf_(int n, int k, double p) {
    p = clamp_probability_(p);
    return std::lgamma(static_cast<double>(n) + 1.0) -
      std::lgamma(static_cast<double>(k) + 1.0) -
      std::lgamma(static_cast<double>(n - k) + 1.0) +
      static_cast<double>(k) * std::log(p) +
      static_cast<double>(n - k) * std::log1p(-p);
  }

  static double log_beta_pdf_(double x, double a, double b) {
    if (!(x > 0.0 && x < 1.0) || !(a > 0.0 && b > 0.0)) {
      return -std::numeric_limits<double>::infinity();
    }
    return (a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) +
      std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
  }

  double beta_(double a, double b) {
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

  static double normal_quantile_(double p) {
    p = clamp_probability_(p);

    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 =  2.506628277459239e+00;

    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;

    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00;
    constexpr double c6 =  2.938163982698783e+00;

    constexpr double d1 =  7.784695709041462e-03;
    constexpr double d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00;
    constexpr double d4 =  3.754408661907416e+00;

    constexpr double plow = 0.02425;
    constexpr double phigh = 1.0 - plow;

    if (p < plow) {
      const double q = std::sqrt(-2.0 * std::log(p));
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    if (p > phigh) {
      const double q = std::sqrt(-2.0 * std::log(1.0 - p));
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    const double q = p - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
      / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  Eigen::VectorXd normal_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

};

} // namespace klhr
