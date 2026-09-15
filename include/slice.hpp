#pragma once

#include "diagonal_metric.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace klhr {

struct SliceOptions {
  std::uint64_t seed = 0;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  double tol = 1e-10;
  // Neal (2003) figure 3, as implemented in his uni.slice: w = 1 and no limit
  // on stepping out or on shrinkage. Zero means unlimited for both counts.
  double initial_width = 1.0;   // Neal's w
  std::size_t max_steps_out = 0;   // Neal's m
  std::size_t max_shrink_steps = 0;
  // No analogue in Neal, whose w is fixed. These bound the metric-scaled
  // width this sampler derives from w instead.
  double min_width = 1e-8;
  double max_width = 1e8;
};

class Slice {
public:
  std::size_t nfev_ = 0;
  // Slice updates do not have MH rejections; this reports successful updates.
  double acceptance_rate_ = 0.0;
  double log_density_ = -std::numeric_limits<double>::infinity();

  Slice(std::string stan_file, std::string json_file,
        const SliceOptions& options = SliceOptions{}) :
    bsm_(std::move(stan_file), std::move(json_file)),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(normalized_options_(options)),
    metric_(bsm_.dim(), opts_.warmup, opts_.windowsize, opts_.windowscale,
            opts_.tol) {

    if (opts_.seed == 0) {
      std::random_device rd;
      const std::uint64_t r1 = rd();
      const std::uint64_t r2 = rd();
      opts_.seed = (r1 << 32) ^ r2;
      if (opts_.seed == 0) {
        opts_.seed = 1;
      }
      rng_.seed(opts_.seed);
    }

    std::uniform_int_distribution<unsigned int> uniform_uint;
    mcmcpp::bsrng bsrng = bsm_.make_rng(uniform_uint(rng_));
    theta_ = bsm_.param_initialize(bsrng);
    if (!theta_.allFinite()) {
      throw std::runtime_error("Slice: invalid initial state");
    }

    log_density_ = bsm_.log_density_noe(theta_);
    ++nfev_;
    if (!std::isfinite(log_density_)) {
      throw std::runtime_error("Slice: initial log density is not finite");
    }
    last_width_ = opts_.initial_width;
  }

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  double width() const {
    return last_width_;
  }

  Eigen::VectorXd metric_variance() const {
    return metric_.variance();
  }

  Eigen::VectorXd draw() {
    ++draw_;
    const Eigen::VectorXd rho = metric_.random_direction(rng_, std_normal_);
    last_width_ = line_width_(rho);
    const bool success = slice_step_(rho, last_width_);
    const double delta = success - acceptance_rate_;
    acceptance_rate_ += delta / draw_;
    metric_.adapt(theta_, draw_);
    return bsm_.param_constrain(theta_);
  }

private:
  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;
  SliceOptions opts_;
  DiagonalMetric metric_;

  Eigen::VectorXd theta_;
  std::size_t draw_ = 0;
  double last_width_ = 1.0;

  static SliceOptions normalized_options_(SliceOptions options) {
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    if (!(options.tol > 0.0) || !std::isfinite(options.tol)) {
      options.tol = 1e-10;
    }
    if (!(options.min_width > 0.0) || !std::isfinite(options.min_width)) {
      options.min_width = 1e-8;
    }
    if (!(options.max_width >= options.min_width) ||
        !std::isfinite(options.max_width)) {
      options.max_width = std::max(1e8, options.min_width);
    }
    if (!(options.initial_width > 0.0) ||
        !std::isfinite(options.initial_width)) {
      options.initial_width = 1.0;
    }
    options.initial_width = std::clamp(
      options.initial_width, options.min_width, options.max_width);
    return options;
  }

  bool slice_step_(const Eigen::VectorXd& rho, const double width) {
    const double u_slice = std::max(
      std_uniform_(rng_), std::numeric_limits<double>::min());
    const double log_slice = log_density_ + std::log(u_slice);

    double left = -std_uniform_(rng_) * width;
    double right = left + width;

    // Stepping out, Neal (2003) figure 3. Unlimited, the two sides expand
    // independently. Limited to m steps, the m-1 available steps are split at
    // random between them, which is what keeps the procedure reversible; m=1
    // therefore means no expansion at all.
    if (opts_.max_steps_out == 0) {
      while (line_log_density_(left, rho) > log_slice) {
        left -= width;
      }
      while (line_log_density_(right, rho) > log_slice) {
        right += width;
      }
    } else {
      std::uniform_int_distribution<std::size_t> left_budget(
        0, opts_.max_steps_out - 1);
      std::size_t steps_left = left_budget(rng_);
      std::size_t steps_right = opts_.max_steps_out - 1 - steps_left;

      while (steps_left > 0 && line_log_density_(left, rho) > log_slice) {
        left -= width;
        --steps_left;
      }
      while (steps_right > 0 && line_log_density_(right, rho) > log_slice) {
        right += width;
        --steps_right;
      }
    }

    // Shrinkage. Neal leaves this unbounded and has no degeneracy test; the
    // one below is ours and does fire, because a bracket can collapse to a
    // single representable point while the slice condition still fails.
    for (std::size_t shrink = 0;
         opts_.max_shrink_steps == 0 || shrink < opts_.max_shrink_steps;
         ++shrink) {
      if (!(left < right)) {
        return false;
      }
      const double t = left + std_uniform_(rng_) * (right - left);
      const double candidate_log_density = line_log_density_(t, rho);
      if (candidate_log_density >= log_slice) {
        theta_ += t * rho;
        log_density_ = candidate_log_density;
        return true;
      }
      if (t < 0.0) {
        left = t;
      } else {
        right = t;
      }
    }
    return false;
  }

  double line_log_density_(const double t,
                           const Eigen::VectorXd& rho) {
    const Eigen::VectorXd candidate = theta_ + t * rho;
    if (!candidate.allFinite()) {
      return -std::numeric_limits<double>::infinity();
    }
    ++nfev_;
    return bsm_.log_density_noe(candidate);
  }

  double line_width_(const Eigen::VectorXd& rho) const {
    const double projected_variance =
      (rho.array().square() * metric_.variance().array()).sum();
    if (!(projected_variance > 0.0) ||
        !std::isfinite(projected_variance)) {
      return opts_.initial_width;
    }
    const double width = opts_.initial_width *
      std::sqrt(projected_variance);
    return std::clamp(width, opts_.min_width, opts_.max_width);
  }

};

} // namespace klhr
