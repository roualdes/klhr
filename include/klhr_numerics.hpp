#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace klhr::numerics {

inline constexpr double log_scale_radius() {
  return 4.6051701859880918; // log(100)
}

inline constexpr double bad_kl_value() {
  return 1e100;
}

inline bool is_bad_kl(const double value) {
  return !(value < bad_kl_value());
}

// Marks an infeasible point: the target was not finite at some quadrature
// node, so there is no usable KL value here. The value is a flat sentinel --
// no additive penalty survives at this magnitude, since the ULP of 1e100 is
// about 1e84 -- so the useful signal lives entirely in the gradient, which
// points back toward the origin of the reparametrised coordinates in every
// component rather than just the scale.
inline void set_bad_kl(const Eigen::VectorXd& eta,
                       double& value,
                       Eigen::VectorXd& grad) {
  value = bad_kl_value();
  grad = Eigen::VectorXd::Zero(eta.size());
  for (Eigen::Index i = 0; i < eta.size(); ++i) {
    if (std::isfinite(eta(i))) {
      grad(i) = eta(i);
    }
  }
}

// Maps an unconstrained real to (-radius, radius), with derivative.
inline double bounded_reparam(const double raw, const double radius) {
  if (!std::isfinite(raw)) {
    return 0.0;
  }
  return radius * std::tanh(raw / radius);
}

inline double bounded_reparam_derivative(const double raw,
                                         const double radius) {
  if (!std::isfinite(raw)) {
    return 0.0;
  }
  const double value = std::tanh(raw / radius);
  return 1.0 - value * value;
}

inline double relative_log_scale(const double raw,
                                 const double log_scale0) {
  if (!std::isfinite(raw)) {
    return log_scale0;
  }
  return log_scale0 + bounded_reparam(raw, log_scale_radius());
}

inline double relative_log_scale_derivative(const double raw) {
  return bounded_reparam_derivative(raw, log_scale_radius());
}

inline double scale_from_log(double log_scale, const double tolerance) {
  if (!std::isfinite(log_scale)) {
    log_scale = 0.0;
  }
  const double max_log =
    std::log(std::numeric_limits<double>::max()) - 2.0;
  const double min_log =
    std::log(std::numeric_limits<double>::min()) + 2.0;
  return std::exp(std::clamp(log_scale, min_log, max_log)) + tolerance;
}

inline double probability_epsilon() {
  return std::numeric_limits<double>::epsilon();
}

inline double clamp_probability(const double probability) {
  const double epsilon = probability_epsilon();
  if (!std::isfinite(probability)) {
    return 0.5;
  }
  return std::clamp(probability, epsilon, 1.0 - epsilon);
}

// True when a CDF value has saturated against the representable range, so
// the overrelaxation kernel would be applied to a clamped surrogate rather
// than the real quantile.
inline bool probability_saturated(const double probability) {
  const double epsilon = probability_epsilon();
  return !std::isfinite(probability) ||
    probability <= epsilon || probability >= 1.0 - epsilon;
}

} // namespace klhr::numerics
