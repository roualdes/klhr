#pragma once

#include <Eigen/Dense>
#include <welford.hpp>
#include <windowedadaptation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace klhr {

// The adapted metric shared by BaseKLHR and Slice: a Welford diagonal
// refreshed at each window close, plus the normalised Gaussian direction law
// built from it. Keeping it in one place is what makes the two samplers
// comparable -- they differ only in what they do along the line they pick,
// and a change here cannot reach one of them without reaching the other.
class DiagonalMetric {
public:
  DiagonalMetric(const Eigen::Index dim,
                 const std::size_t warmup,
                 const std::size_t windowsize,
                 const std::size_t windowscale,
                 const double tol) :
    dim_(dim),
    warmup_(warmup),
    tol_(tol),
    windowed_adaptation_(warmup, windowsize, windowscale),
    online_moments_(dim),
    mean_(Eigen::VectorXd::Zero(dim)) {
    set_variance_(Eigen::VectorXd::Ones(dim));
  }

  const Eigen::VectorXd& variance() const { return variance_; }

  const Eigen::VectorXd& mean() const { return mean_; }

  // Seed the mean from the initial draw rather than the origin. BaseKLHR
  // anchors its canonical line origin here, and on a badly scaled posterior
  // the origin can be many standard deviations from anywhere the chain will
  // ever be. Slice has no such anchor and does not call this.
  void seed_mean(const Eigen::VectorXd& theta) {
    mean_ = theta;
  }

  void adapt(const Eigen::VectorXd& theta, const std::size_t draw) {
    if (draw > warmup_) {
      return;
    }

    online_moments_.update(theta);

    if (windowed_adaptation_.window_closed(draw)) {
      mean_ = online_moments_.mean();
      set_variance_(online_moments_.variance());
      online_moments_.reset();
    }
  }

  // A Gaussian shaped like the diagonal, normalised. nu is symmetric and does
  // not depend on theta, so nu(d)/nu(-d) is one and the direction never
  // enters a Hastings ratio.
  template <typename Rng, typename Normal>
  Eigen::VectorXd random_direction(Rng& rng, Normal& std_normal) const {
    Eigen::VectorXd rho = variance_.array().sqrt().matrix().cwiseProduct(
      normal_rng(dim_, rng, std_normal));

    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= tol_) {
      rho = normal_rng(dim_, rng, std_normal);
      norm = rho.norm();
    }

    rho /= norm;
    return rho;
  }

  template <typename Rng, typename Normal>
  static Eigen::VectorXd normal_rng(const Eigen::Index D, Rng& rng,
                                    Normal& std_normal) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D,
                  [&]() { return std_normal(rng); });
    return out;
  }

private:
  Eigen::Index dim_;
  std::size_t warmup_;
  double tol_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  Eigen::VectorXd mean_;
  Eigen::VectorXd variance_;

  // The variance moves only at a window close and at construction, so it is
  // sanitised at those two points rather than on every read. The read side is
  // hot -- the direction draw wants it once per draw, and sanitising costs an
  // allocation plus an nth_element over D.
  void set_variance_(Eigen::VectorXd variance) {
    variance_ = sanitize_(std::move(variance));
  }

  // Floor relative to a typical variance, not absolutely. A coordinate that
  // happens not to move inside a window can report a variance many orders
  // below the rest; an absolute floor of tol lets it through, and the
  // direction draw is then dominated by that one coordinate.
  Eigen::VectorXd sanitize_(Eigen::VectorXd variance) const {
    std::vector<double> positive;
    positive.reserve(static_cast<std::size_t>(variance.size()));
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (std::isfinite(variance(d)) && variance(d) > 0.0) {
        positive.push_back(variance(d));
      }
    }
    double typical = 1.0;
    if (!positive.empty()) {
      const auto middle = positive.begin() + positive.size() / 2;
      std::nth_element(positive.begin(), middle, positive.end());
      typical = *middle;
    }
    const double floor = std::max(tol_, 1e-6 * typical);
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (!std::isfinite(variance(d)) || variance(d) <= 0.0) {
        variance(d) = std::max(typical, floor);
      } else {
        variance(d) = std::max(variance(d), floor);
      }
    }
    return variance;
  }
};

} // namespace klhr
