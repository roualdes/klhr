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

inline void set_bad_kl(const Eigen::VectorXd& eta,
                       double& value,
                       Eigen::VectorXd& grad) {
  value = bad_kl_value();
  grad = Eigen::VectorXd::Zero(eta.size());
  if (eta.size() > 1 && std::isfinite(eta(1))) {
    grad(1) = eta(1);
  }
}

inline double relative_log_scale(const double raw,
                                 const double log_scale0) {
  const double radius = log_scale_radius();
  if (!std::isfinite(raw)) {
    return log_scale0;
  }
  return log_scale0 + radius * std::tanh(raw / radius);
}

inline double relative_log_scale_derivative(const double raw) {
  const double radius = log_scale_radius();
  if (!std::isfinite(raw)) {
    return 0.0;
  }
  const double value = std::tanh(raw / radius);
  return 1.0 - value * value;
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

inline double clamp_probability(const double probability) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  if (!std::isfinite(probability)) {
    return 0.5;
  }
  return std::clamp(probability, epsilon, 1.0 - epsilon);
}

} // namespace klhr::numerics
