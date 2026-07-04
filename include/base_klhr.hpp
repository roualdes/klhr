#pragma once

#include "gausshermite.hpp"
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
  Eigen::Index J = 4;
  double l = 0.0;
  std::size_t initial_transport_steps = 0;
  std::size_t transport_max_reflections = 128;
  double transport_initial_distance = 1.0;
  double transport_min_distance = 1e-8;
  double transport_max_distance = 1e6;
  double transport_max_logp_drop = 1000.0;
  double transport_direction_persistence = 0.9;
  double transport_failure_direction_decay = 0.5;
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
    opts_(options),
    windowed_adaptation_(options.warmup, options.windowsize,
                         options.windowscale),
    online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), options.J, options.l, options.tol),
    transport_distance_(options.transport_initial_distance) {

    if (opts_.seed == 0) {
      std::random_device rd;
      rng_.seed(rd());
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
    grad_ = Eigen::VectorXd::Zero(D);
    transport_direction_state_ = normal_rng_(D);
    regularize_transport_direction_();

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    bsm_.log_density_gradient_noe(theta_, log_density_, grad_);
    ++nfev_;
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

  Eigen::VectorXd draw() {
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      initial_transport_step_();
      adapt_warmup_(theta_, draw_);
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

  Eigen::VectorXd random_direction() {
    Eigen::VectorXd weights = eigvals_.cwiseMax(0.0);
    std::discrete_distribution<Eigen::Index> component(weights.data(),
                                                      weights.data() + weights.size());
    const Eigen::Index j = component(rng_);

    Eigen::VectorXd rho(dim());
    const Eigen::VectorXd center = eigvecs_.col(j);
    for (Eigen::Index d = 0; d < rho.size(); ++d) {
      const double variance = std::max(cov_(d), opts_.tol);
      rho(d) = center(d) + std::sqrt(variance) * std_normal_(rng_);
    }

    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_rng_(dim());
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
    (void) center;
    (void) rho;
    return Eigen::VectorXd::Constant(2, nan_());
  }

  virtual double transport_distance_proposal_(const Eigen::VectorXd& eta) {
    (void) eta;
    return nan_();
  }

  virtual void record_kl_step_(const Eigen::VectorXd& eta, const double xi,
                               const bool accepted) {
    (void) eta;
    (void) xi;
    (void) accepted;
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  OnlinePCA online_pca_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
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
  double transport_distance_;

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

      const Eigen::VectorXd delta = next.theta - theta0;
      Eigen::VectorXd scaled_delta = delta.array() / scale.array();
      if (!scaled_delta.allFinite()) {
        failed = true;
        break;
      }
      const double turning = scaled_delta.dot(direction);
      const double initial_turning = scaled_delta.dot(direction0);
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

  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    if (adaptation_draw == 0 || adaptation_draw > opts_.warmup) {
      return;
    }

    online_moments_.update(theta);
    online_pca_.update(theta - mean_);

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();

      if (opts_.J > 0) {
        eigvecs_.leftCols(opts_.J) = online_pca_.vectors();
        eigvals_.head(opts_.J) = online_pca_.values();
      }
      online_pca_.reset();
    }
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
