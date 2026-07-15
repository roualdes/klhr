#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace klhr {

class Adam {
public:
  Adam(const double learning_rate = 0.001,
       const double beta1 = 0.9,
       const double beta2 = 0.999,
       const double epsilon = 1e-8) :
    learning_rate_(positive_or_(learning_rate, 0.001)),
    beta1_(beta_or_(beta1, 0.9)),
    beta2_(beta_or_(beta2, 0.999)),
    epsilon_(positive_or_(epsilon, 1e-8)) {}

  double step(const double gradient) {
    ++t_;
    m_ = beta1_ * m_ + (1.0 - beta1_) * gradient;
    v_ = beta2_ * v_ + (1.0 - beta2_) * gradient * gradient;

    const double m_hat = m_ / (1.0 - std::pow(beta1_, t_));
    const double v_hat = v_ / (1.0 - std::pow(beta2_, t_));
    return learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
  }

  void reset() {
    m_ = 0.0;
    v_ = 0.0;
    t_ = 0;
  }

private:
  static double positive_or_(const double value, const double fallback) {
    return value > 0.0 && std::isfinite(value) ? value : fallback;
  }

  static double beta_or_(const double value, const double fallback) {
    const double beta = std::isfinite(value) ? value : fallback;
    return std::clamp(beta, 0.0,
                      std::nextafter(1.0, 0.0));
  }

  double learning_rate_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double m_ = 0.0;
  double v_ = 0.0;
  std::size_t t_ = 0;
};

} // namespace klhr
