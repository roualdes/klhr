#pragma once

#include <cmath>
#include <cstddef>

namespace klhr {

class Adam {
public:
  Adam(const double learning_rate = 0.001,
       const double beta1 = 0.9,
       const double beta2 = 0.999,
       const double epsilon = 1e-8) :
    learning_rate_(learning_rate),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(epsilon) {}

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
  double learning_rate_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double m_ = 0.0;
  double v_ = 0.0;
  std::size_t t_ = 0;
};

} // namespace klhr
