#pragma once

#include "bfgs.hpp"
#include "gausslaguerre.hpp"
#include "klhr_numerics.hpp"
#include "onlinepca.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>
#include <welford.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace klhr {

struct ReflectedTransportOptions {
  // TODO: rename to N, to match naming in BaseKLHR
  Eigen::Index quadrature_size = 8;
  Eigen::Index pca_rank = 1;
  double tol = 1e-10;
  double grad_clip = std::numeric_limits<double>::infinity();
  double gtol = 1e-3;
  double pca_l = 0.0;
  double covariance_shrink = 0.25;
  double covariance_ratio_cap = 4.0;
  std::size_t max_reflections = 500;
  double initial_distance = 1.0;
  double min_distance = 1e-8;
  double max_distance = 1e6;
  double max_logp_drop = 1000.0;
  double max_segment_logp_drop = 1000.0;
  double max_endpoint_from_best_drop = 100.0;
  double direction_persistence = 0.9;
  double failure_direction_decay = 0.25;
};

class ReflectedTransport {
public:
  struct State {
    Eigen::VectorXd theta;
    Eigen::VectorXd grad;
    double log_density = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
  };

  struct StepResult {
    State state;
    bool moved = false;
    std::size_t evaluations = 0;
  };

  struct Handoff {
    State state;
    Eigen::VectorXd covariance;
    Eigen::MatrixXd pca_basis;
    Eigen::VectorXd pca_weights;
    bool pca_ready = false;
    bool pca_whitened = false;
    bool rollback = false;
  };

  ReflectedTransport(const Eigen::Index dim,
                     ReflectedTransportOptions options) :
    dim_(checked_dimension_(dim)),
    opts_(normalize_options_(std::move(options), dim_)),
    moments_(dim_),
    pca_(dim_, opts_.pca_rank, opts_.pca_l, opts_.tol),
    covariance_(Eigen::VectorXd::Ones(dim_)),
    smooth_mean_(Eigen::VectorXd::Zero(dim_)),
    mean_direction_basis_(Eigen::MatrixXd::Zero(dim_, opts_.pca_rank)),
    mean_direction_weights_(Eigen::VectorXd::Ones(opts_.pca_rank)),
    handoff_basis_(Eigen::MatrixXd::Zero(dim_, opts_.pca_rank)),
    handoff_weights_(Eigen::VectorXd::Zero(opts_.pca_rank)),
    distance_(opts_.initial_distance) {
    gauss_laguerre(opts_.quadrature_size, laguerre_weights_, laguerre_nodes_);
    if (!laguerre_weights_.allFinite() || !laguerre_nodes_.allFinite() ||
        (laguerre_nodes_.array() <= 0.0).any()) {
      throw std::runtime_error("invalid Gauss-Laguerre quadrature");
    }
  }

  void initialize(const State& initial,
                  mcmcpp::rng& rng,
                  std::normal_distribution<double>& standard_normal) {
    const bool valid_initial = initial.valid &&
      initial.theta.size() == dim_ && initial.grad.size() == dim_ &&
      initial.theta.allFinite() && initial.grad.allFinite() &&
      std::isfinite(initial.log_density);
    state_ = initial;
    state_.valid = valid_initial;
    initial_state_ = initial;
    initial_state_.valid = valid_initial;
    best_state_ = initial;
    best_state_.valid = valid_initial;
    covariance_.setOnes();
    smooth_mean_ = valid_initial ? initial.theta :
      Eigen::VectorXd::Zero(dim_);
    moments_.reset();
    if (valid_initial) {
      moments_.update(initial.theta);
    }
    pca_.reset();
    mean_direction_basis_.setZero();
    mean_direction_weights_.setOnes();
    mean_direction_ready_ = false;
    handoff_basis_.setZero();
    handoff_weights_.setZero();
    handoff_ready_ = false;
    handoff_whitened_ = false;
    rollback_ = false;
    distance_ = opts_.initial_distance;
    direction_ = normal_rng_(dim_, rng, standard_normal);
    regularize_direction_(rng, standard_normal);
    initialized_ = valid_initial;
  }

  template <typename Constrain>
  StepResult step(mcmcpp::bsmodel& model,
                  mcmcpp::rng& rng,
                  std::uniform_real_distribution<double>& standard_uniform,
                  std::normal_distribution<double>& standard_normal,
                  Constrain&& constrain) {
    StepResult result;
    result.state = state_;

    if (!initialized_) {
      return result;
    }

    const State start = state_;
    const Eigen::VectorXd scale = metric_scale_();
    partial_refresh_direction_(rng, standard_normal);
    const Eigen::VectorXd direction0 =
      normalized_direction_(rng, standard_normal);

    // TODO: are all these checks necessary on every step?
    if (!start.theta.allFinite() || !start.grad.allFinite() ||
        !scale.allFinite() || !std::isfinite(start.log_density) ||
        !direction0.allFinite()) {
      regularize_direction_(rng, standard_normal);
      result.state = state_;
      return result;
    }

    State current = start;
    State endpoint = current;
    Eigen::VectorXd direction = direction0;
    Eigen::VectorXd endpoint_direction = direction0;
    bool moved = false;
    bool failed = false;

    for (std::size_t reflection = 0;
         reflection < opts_.max_reflections; ++reflection) {
      const Eigen::VectorXd rho =
        (scale.array() * direction.array()).matrix();
      Eigen::VectorXd eta = fit_ray_(model, current.theta, rho,
                                     result.evaluations);
      double distance = distance_proposal_(eta, rng, standard_uniform);
      // EAR: check distance_proposal_, do these checks happen internally?
      if (!eta.allFinite() || !(distance > 0.0) ||
          !std::isfinite(distance)) {
        failed = true;
        break;
      }
      distance = std::clamp(distance, opts_.min_distance, opts_.max_distance);

      State next = ray_step_(model, current, distance, rho,
                             result.evaluations);
      if (!next.valid) {
        failed = true;
        break;
      }

      if (start.log_density - next.log_density >
          opts_.max_logp_drop) {
        failed = true;
        break;
      }
      if (current.log_density - next.log_density >
          opts_.max_segment_logp_drop) {
        break;
      }

      const Eigen::VectorXd delta = next.theta - start.theta;
      if (!delta.allFinite()) {
        failed = true;
        break;
      }
      const double turning = delta.dot(direction);
      const double initial_turning = delta.dot(direction0);
      if (!std::isfinite(turning) || !std::isfinite(initial_turning) ||
          turning <= 0.0 || initial_turning <= 0.0) {
        break;
      }

      // TODO: will you double check why we have endpoint and current?
      // when do their values differ?
      endpoint = next;
      current = next;
      moved = true;
      distance_ = distance;

      const Eigen::VectorXd reflected =
        reflected_direction_(direction, endpoint.grad, scale);
      if (!reflected.allFinite()) {
        break;
      }
      direction = reflected;
      endpoint_direction = direction;
    }

    if (moved) {
      // TODO: can it be the case that endoint.theta is finite
      // but constrain(endpoint.theta) is not?
      // If this isn't likely, then I'd just check endpoint.theta
      // for being finite, and in fact that might already happen
      // when next is set, thus negating this check for finite-ness entirely.
      Eigen::VectorXd proposal = constrain(endpoint.theta);
      if (!proposal.allFinite()) {
        moved = false;
      } else {
        state_ = endpoint;
        update_covariance_(state_.theta);
        update_best_state_();
      }
    }

    update_direction_(moved, failed, endpoint_direction, direction0,
                      rng, standard_normal);

    result.state = state_;
    result.moved = moved;
    return result;
  }

  Handoff finish(mcmcpp::rng& rng,
                 std::normal_distribution<double>& standard_normal) {
    const double endpoint_from_best_drop =
      best_state_.log_density - state_.log_density;
    if (std::isfinite(endpoint_from_best_drop) &&
        endpoint_from_best_drop > opts_.max_endpoint_from_best_drop) {
      rollback_to_initial_(rng, standard_normal);
      return handoff_();
    }

    handoff_ready_ = set_mean_direction_from_pca_();
    handoff_whitened_ = handoff_ready_;
    if (handoff_ready_) {
      handoff_basis_ = mean_direction_basis_;
      handoff_weights_ = mean_direction_weights_;
    } else {
      handoff_basis_.setZero();
      handoff_weights_.setZero();
    }
    pca_.reset();
    return handoff_();
  }

private:
  static Eigen::Index checked_dimension_(const Eigen::Index dim) {
    if (dim < 0) {
      throw std::invalid_argument(
        "ReflectedTransport: dimension must be nonnegative");
    }
    return dim;
  }

  static ReflectedTransportOptions normalize_options_(
      ReflectedTransportOptions options, const Eigen::Index dim) {
    options.quadrature_size =
      std::max<Eigen::Index>(1, options.quadrature_size);
    options.pca_rank = std::clamp(options.pca_rank, Eigen::Index{0}, dim);
    if (!(options.tol > 0.0) || !std::isfinite(options.tol)) {
      options.tol = 1e-10;
    }
    if (std::isfinite(options.grad_clip)) {
      options.grad_clip = std::abs(options.grad_clip);
    } else {
      options.grad_clip = std::numeric_limits<double>::infinity();
    }
    if (!(options.gtol > 0.0) || !std::isfinite(options.gtol)) {
      options.gtol = 1e-3;
    }
    if (!std::isfinite(options.pca_l)) {
      options.pca_l = 0.0;
    }
    options.covariance_shrink =
      std::isfinite(options.covariance_shrink) ?
      std::clamp(options.covariance_shrink, 0.0, 1.0) : 0.25;
    options.covariance_ratio_cap =
      std::isfinite(options.covariance_ratio_cap) ?
      std::max(options.covariance_ratio_cap, 1.0) : 4.0;
    if (!(options.min_distance > 0.0) ||
        !std::isfinite(options.min_distance)) {
      options.min_distance = 1e-8;
    }
    if (!(options.max_distance >= options.min_distance) ||
        !std::isfinite(options.max_distance)) {
      options.max_distance = std::max(1e6, options.min_distance);
    }
    if (!(options.initial_distance > 0.0) ||
        !std::isfinite(options.initial_distance)) {
      options.initial_distance = 1.0;
    }
    options.initial_distance = std::clamp(
      options.initial_distance, options.min_distance, options.max_distance);
    options.max_logp_drop = std::isfinite(options.max_logp_drop) ?
      std::abs(options.max_logp_drop) :
      std::numeric_limits<double>::infinity();
    options.max_segment_logp_drop =
      std::isfinite(options.max_segment_logp_drop) ?
      std::abs(options.max_segment_logp_drop) :
      std::numeric_limits<double>::infinity();
    if (std::isfinite(options.max_endpoint_from_best_drop)) {
      options.max_endpoint_from_best_drop =
        std::max(0.0, options.max_endpoint_from_best_drop);
    } else {
      options.max_endpoint_from_best_drop =
        std::numeric_limits<double>::infinity();
    }
    options.direction_persistence =
      std::isfinite(options.direction_persistence) ?
      std::clamp(options.direction_persistence, 0.0, 1.0) : 0.9;
    options.failure_direction_decay =
      std::isfinite(options.failure_direction_decay) ?
      std::clamp(options.failure_direction_decay, 0.0, 1.0) : 0.25;
    return options;
  }

  Eigen::VectorXd fit_ray_(mcmcpp::bsmodel& model,
                           const Eigen::VectorXd& center,
                           const Eigen::VectorXd& rho,
                           std::size_t& evaluations) {
    double distance0 = distance_;
    if (!(distance0 >= opts_.min_distance) ||
        !(distance0 <= opts_.max_distance) || !std::isfinite(distance0)) {
      distance0 = opts_.initial_distance;
    }
    const double log_scale0 = std::log(distance0);

    Eigen::VectorXd init(2);
    init << 0.0, 0.0;
    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      weibull_kl_(model, eta, center, rho, log_scale0, value, grad);
    };
    bfgs::BfgsResult result =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 4});
    // TODO: instead of laguerre_nodes_.size(), let's use opts_.N
    evaluations += result.nfev * laguerre_nodes_.size();

    // TODO: no need to check the size, it will always be 2.
    // TODO: will result.x every be non-finite? At worst, bad_kl_value
    // will be returned, which is finite. Can't we ditch that check too?
    const Eigen::VectorXd raw =
      result.x.size() == 2 && result.x.allFinite() ? result.x : init;
    Eigen::VectorXd out(2);
    out << bounded_log_shape_(raw(0)),
      relative_log_scale_(raw(1), log_scale0);
    return out;
  }

  double distance_proposal_(const Eigen::VectorXd& eta,
                            mcmcpp::rng& rng,
                            std::uniform_real_distribution<double>&
                              standard_uniform) const {
    if (!eta.allFinite()) {
      return nan_();
    }
    const double shape = scale_from_log_(eta(0));
    const double scale = scale_from_log_(eta(1));
    if (!(shape > 0.0) || !(scale > 0.0) ||
        !std::isfinite(shape) || !std::isfinite(scale)) {
      return nan_();
    }
    const double u = clamp_probability_(standard_uniform(rng));
    const double x = -std::log1p(-u);
    const double distance = scale * std::pow(x, 1.0 / shape);
    if (!(distance > 0.0) || !std::isfinite(distance)) {
      return nan_();
    }
    return distance;
  }

  State ray_step_(mcmcpp::bsmodel& model,
                  const State& state,
                  const double distance,
                  const Eigen::VectorXd& rho,
                  std::size_t& evaluations) const {
    State out;
    out.theta = Eigen::VectorXd::Constant(dim_, nan_());
    out.grad = Eigen::VectorXd::Constant(dim_, nan_());
    // TODO: are all these checks necessary each ray_step_?
    if (!state.valid || !(distance > 0.0) || !std::isfinite(distance) ||
        !state.theta.allFinite() || !state.grad.allFinite() ||
        !rho.allFinite()) {
      return out;
    }

    out.theta = state.theta + distance * rho;
    if (!out.theta.allFinite()) {
      return out;
    }
    model.log_density_gradient_noe(out.theta, out.log_density, out.grad);
    ++evaluations;
    if (!std::isfinite(out.log_density) || !out.grad.allFinite()) {
      return out;
    }
    out.valid = true;
    return out;
  }

  void weibull_kl_(mcmcpp::bsmodel& model,
                   const Eigen::VectorXd& eta,
                   const Eigen::VectorXd& center,
                   const Eigen::VectorXd& rho,
                   const double log_scale0,
                   double& value,
                   Eigen::VectorXd& grad) const {
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
    Eigen::VectorXd xi(dim_);
    Eigen::VectorXd grad_logp(dim_);
    const double inv_shape = 1.0 / shape;
    for (Eigen::Index n = 0; n < laguerre_nodes_.size(); ++n) {
      const double xn = laguerre_nodes_(n);
      const double wn = laguerre_weights_(n);
      const double log_x = std::log(xn);
      const double x_power =
        std::exp(std::clamp(inv_shape * log_x, -700.0, 700.0));
      const double distance = scale * x_power;
      xi = center + distance * rho;
      // TODO: shouldn't these checks be on xi, since that's what's going into
      // model.log_density_gradient_noe?
      // But in general, these are good checks because they might prevent
      // log density gradient calculations
      if (!(distance > 0.0) || !std::isfinite(distance) || !xi.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      double logp;
      model.log_density_gradient_noe(xi, logp, grad_logp);

      // TODO: necessary checks?
      if (!std::isfinite(logp) || !grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      grad_logp =
        grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      // TODO: necessary checks?
      if (!grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      const double line_grad = grad_logp.dot(rho);
      // TOOD: doesn't this one check encapsulate the previous two checks,
      // since it takes into account rho and grad_logp?
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
      grad(1) += wn * (-1.0 -
                       line_grad * d_distance_d_log_scale);
    }
    grad(0) *= dlog_shape;
    grad(1) *= dlog_scale;
  }

  Eigen::VectorXd reflected_direction_(const Eigen::VectorXd& direction,
                                       const Eigen::VectorXd& grad,
                                       const Eigen::VectorXd& scale) const {
    // TOOD: it seems like we are checking direction in every method?
    // maybe I'm imagining this. But can't we just check it after it's produced,
    // or after it is reflected and then be done with the extra checks?
    if (!direction.allFinite() || !grad.allFinite() || !scale.allFinite()) {
      return Eigen::VectorXd::Constant(dim_, nan_());
    }
    Eigen::VectorXd normal = (scale.array() * grad.array()).matrix();
    const double normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= opts_.tol) {
      return Eigen::VectorXd::Constant(dim_, nan_());
    }
    normal /= normal_norm;
    Eigen::VectorXd reflected =
      direction - 2.0 * direction.dot(normal) * normal;
    const double reflected_norm = reflected.norm();
    // TODO: doesn't this check encapsulate the check before it,
    // thus making the check before this obsolete?
    if (!std::isfinite(reflected_norm) || reflected_norm <= opts_.tol) {
      return Eigen::VectorXd::Constant(dim_, nan_());
    }
    return reflected / reflected_norm;
  }

  void partial_refresh_direction_(
      mcmcpp::rng& rng,
      std::normal_distribution<double>& standard_normal) {
    const double a = opts_.direction_persistence;
    const double b = std::sqrt(std::max(0.0, 1.0 - a * a));
    if (!direction_.allFinite()) {
      direction_ = normal_rng_(dim_, rng, standard_normal);
    } else {
      direction_ = a * direction_ +
        b * normal_rng_(dim_, rng, standard_normal);
    }
    regularize_direction_(rng, standard_normal);
  }

  void update_direction_(const bool moved,
                         const bool failed,
                         const Eigen::VectorXd& endpoint_direction,
                         const Eigen::VectorXd& initial_direction,
                         mcmcpp::rng& rng,
                         std::normal_distribution<double>& standard_normal) {
    const double decay = opts_.failure_direction_decay;
    if (moved && endpoint_direction.allFinite()) {
      direction_ = endpoint_direction;
      if (failed) {
        direction_ *= decay;
      }
    } else {
      direction_ = -decay * initial_direction;
    }
    regularize_direction_(rng, standard_normal);
  }

  void regularize_direction_(
      mcmcpp::rng& rng,
      std::normal_distribution<double>& standard_normal) {
    if (!direction_.allFinite()) {
      direction_ = normal_rng_(dim_, rng, standard_normal);
    }
    // TODO: Can't we just check the norm for finite-ness?
    // Doesn't that imply allFinite()? Such that this if statement
    // will make useless the previous if statement?
    const double norm = direction_.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      direction_ = normal_rng_(dim_, rng, standard_normal);
    }
  }

  // TODO: do we really need both normalized_direction_ and regularize_direction_?
  // Can't these be brought together?
  Eigen::VectorXd normalized_direction_(
      mcmcpp::rng& rng,
      std::normal_distribution<double>& standard_normal) {
    regularize_direction_(rng, standard_normal);
    const double norm = direction_.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      return Eigen::VectorXd::Constant(dim_, nan_());
    }
    return direction_ / norm;
  }

  // TODO: this looks like it's the same thing as
  // base_klhr.hpp's diagonal_variance_()
  // if so, they should be identical in name and form, right?
  Eigen::VectorXd metric_scale_() const {
    Eigen::VectorXd scale(dim_);
    for (Eigen::Index d = 0; d < dim_; ++d) {
      double variance = covariance_(d);
      if (!std::isfinite(variance) || variance <= opts_.tol) {
        variance = 1.0;
      }
      scale(d) = std::sqrt(variance);
    }
    return scale;
  }

  void update_covariance_(const Eigen::VectorXd& theta) {
    if (!theta.allFinite()) {
      return;
    }
    moments_.update(theta);
    if (moments_.count() <= 1) {
      return;
    }

    const Eigen::VectorXd raw_mean = moments_.mean();
    const Eigen::VectorXd raw_variance = moments_.variance();
    if (!raw_mean.allFinite() || !raw_variance.allFinite()) {
      return;
    }

    if (!covariance_.allFinite()) {
      covariance_ = Eigen::VectorXd::Ones(dim_);
    }
    for (Eigen::Index d = 0; d < dim_; ++d) {
      const double old_variance = std::max(covariance_(d), opts_.tol);
      const double target_variance =
        std::max(raw_variance(d), opts_.tol);
      const double proposal =
        (1.0 - opts_.covariance_shrink) * old_variance +
        opts_.covariance_shrink * target_variance;
      const double lo = old_variance / opts_.covariance_ratio_cap;
      const double hi = old_variance * opts_.covariance_ratio_cap;
      covariance_(d) = std::clamp(proposal, lo, hi);
    }

    update_smooth_mean_(raw_mean);
    update_mean_direction_basis_(theta);
  }

  void update_smooth_mean_(const Eigen::VectorXd& raw_mean) {
    if (!raw_mean.allFinite()) {
      return;
    }
    if (!smooth_mean_.allFinite()) {
      smooth_mean_ = raw_mean;
      return;
    }

    for (Eigen::Index d = 0; d < dim_; ++d) {
      const double variance = covariance_(d);
      const double sd = std::sqrt(std::max(variance, opts_.tol));
      const double max_step = opts_.covariance_ratio_cap * sd;
      const double old_mean = smooth_mean_(d);
      const double proposal = old_mean +
        opts_.covariance_shrink * (raw_mean(d) - old_mean);
      if (!std::isfinite(proposal) || !std::isfinite(max_step)) {
        continue;
      }
      smooth_mean_(d) =
        std::clamp(proposal, old_mean - max_step, old_mean + max_step);
    }
  }

  void update_mean_direction_basis_(const Eigen::VectorXd& theta) {
    if (opts_.pca_rank <= 0 || !theta.allFinite() ||
        moments_.count() < 2 || !smooth_mean_.allFinite()) {
      return;
    }
    const Eigen::VectorXd centered = theta - smooth_mean_;
    if (!centered.allFinite() || centered.norm() <= opts_.tol) {
      return;
    }
    pca_.update(centered);
    (void) set_mean_direction_from_pca_();
  }

  bool set_mean_direction_from_pca_() {
    if (opts_.pca_rank <= 0 || pca_.count() < opts_.pca_rank) {
      return false;
    }
    const Eigen::MatrixXd basis = pca_.vectors();
    const Eigen::VectorXd weights = pca_.values();
    if (!basis.allFinite() || !weights.allFinite()) {
      return false;
    }

    mean_direction_basis_ = basis.leftCols(opts_.pca_rank);
    mean_direction_weights_ =
      weights.head(opts_.pca_rank).cwiseMax(opts_.tol);
    for (Eigen::Index j = 0; j < mean_direction_basis_.cols(); ++j) {
      const double norm = mean_direction_basis_.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        mean_direction_ready_ = false;
        return false;
      }
      mean_direction_basis_.col(j) /= norm;
    }
    mean_direction_ready_ = true;
    return true;
  }

  void update_best_state_() {
    if (!state_.theta.allFinite() || !state_.grad.allFinite() ||
        !std::isfinite(state_.log_density)) {
      return;
    }
    if (!std::isfinite(best_state_.log_density) ||
        state_.log_density > best_state_.log_density) {
      best_state_ = state_;
    }
  }

  void rollback_to_initial_(
      mcmcpp::rng& rng,
      std::normal_distribution<double>& standard_normal) {
    if (initial_state_.theta.allFinite() &&
        initial_state_.grad.allFinite() &&
        std::isfinite(initial_state_.log_density)) {
      state_ = initial_state_;
    }
    rollback_ = true;
    covariance_.setOnes();
    smooth_mean_ = state_.theta;
    moments_.reset();
    if (state_.theta.allFinite()) {
      moments_.update(state_.theta);
    }
    pca_.reset();
    mean_direction_basis_.setZero();
    mean_direction_weights_.setOnes();
    mean_direction_ready_ = false;
    handoff_basis_.setZero();
    handoff_weights_.setZero();
    handoff_ready_ = false;
    handoff_whitened_ = false;
    distance_ = opts_.initial_distance;
    direction_ = normal_rng_(dim_, rng, standard_normal);
    regularize_direction_(rng, standard_normal);
  }

  Handoff handoff_() const {
    return {
      .state = state_,
      .covariance = covariance_,
      .pca_basis = handoff_basis_,
      .pca_weights = handoff_weights_,
      .pca_ready = handoff_ready_,
      .pca_whitened = handoff_whitened_,
      .rollback = rollback_,
    };
  }

  void set_bad_kl_(const Eigen::VectorXd& eta,
                   double& value,
                   Eigen::VectorXd& grad) const {
    numerics::set_bad_kl(eta, value, grad);
  }

  double relative_log_scale_(const double raw,
                             const double log_scale0) const {
    return numerics::relative_log_scale(raw, log_scale0);
  }

  static double relative_log_scale_derivative_(const double raw) {
    return numerics::relative_log_scale_derivative(raw);
  }

  double scale_from_log_(double log_scale) const {
    return numerics::scale_from_log(log_scale, opts_.tol);
  }

  static double bounded_log_shape_(const double raw) {
    constexpr double radius = 2.3025850929940457;
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    return radius * std::tanh(raw / radius);
  }

  static double bounded_log_shape_derivative_(const double raw) {
    constexpr double radius = 2.3025850929940457;
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    const double value = std::tanh(raw / radius);
    return 1.0 - value * value;
  }

  static double clamp_probability_(const double probability) {
    return numerics::clamp_probability(probability);
  }

  static Eigen::VectorXd normal_rng_(
      const Eigen::Index dim,
      mcmcpp::rng& rng,
      std::normal_distribution<double>& standard_normal) {
    Eigen::VectorXd out(dim);
    std::generate(out.data(), out.data() + dim,
                  [&]() { return standard_normal(rng); });
    return out;
  }

  static constexpr double nan_() {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Eigen::Index dim_;
  ReflectedTransportOptions opts_;
  mcmcpp::WelfordAccumulator moments_;
  OnlinePCA pca_;
  Eigen::VectorXd laguerre_weights_;
  Eigen::VectorXd laguerre_nodes_;
  State state_;
  State initial_state_;
  State best_state_;
  Eigen::VectorXd covariance_;
  Eigen::VectorXd smooth_mean_;
  Eigen::VectorXd direction_;
  Eigen::MatrixXd mean_direction_basis_;
  Eigen::VectorXd mean_direction_weights_;
  Eigen::MatrixXd handoff_basis_;
  Eigen::VectorXd handoff_weights_;
  double distance_;
  bool initialized_ = false;
  bool mean_direction_ready_ = false;
  bool handoff_ready_ = false;
  bool handoff_whitened_ = false;
  bool rollback_ = false;
};

} // namespace klhr
