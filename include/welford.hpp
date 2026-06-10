#pragma once

#include <cstddef>
#include <Eigen/Dense>
#include <limits>

/**
 * @brief Welford accumulator for calculating online/streaming mean
 * and (sample) variances.
 */
class WelfordAccumulator {
public:
  /**
   * @brief Initialize with zero observations.
   */
  WelfordAccumulator(std::size_t D) : D_(D) {
    m_.resize(D);
    v_.resize(D);
  }

  /**
   * @brief Update mean, variance, and number of observations.
   *
   * @param[in] x the observation.
   */
  void update(const Eigen::Ref<const Eigen::VectorXd>& x) {
    ++n_;
    const Eigen::VectorXd d = x - m_;
    const double w = 1 / static_cast<double>(n_);
    m_ += d * w;
    v_ += -v_ * w + d.array().square().matrix() * w * (1 - w);
  }

  /**
   * @brief Return the (scalar) mean of observations.
   *
   * @return The mean, so far.
   */
  Eigen::VectorXd mean() const {
    return m_;
  }

  /**
   * @brief Return the sample variance of observations.
   *
   * @return The variance, so far.
   */
  Eigen::VectorXd variance() const {
    const double N = static_cast<double>(n_);
    if (n_ > 1) {
      return v_ * N / (N - 1);
    }
    return Eigen::VectorXd::Constant(D_, std::numeric_limits<double>::quiet_NaN());
  }

  /**
   * @brief Return the number of observations.
   *
   * @return The number of observations, so far.
   */
  std::size_t count() const {
    return n_;
  }

  /**
   * @brief Reset the mean, variance, and number of observations.
   */
  void reset() {
    n_ = 0;
    m_.setZero();
    v_.setZero();
  }

private:
  std::size_t D_;
  std::size_t n_;
  Eigen::VectorXd m_;
  Eigen::VectorXd v_;
};
