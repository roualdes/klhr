#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace klhr::detail {

template <typename Options>
Options normalize_gradient_sampler_options(Options options,
                                           const double default_target) {
  options.windowsize = std::max<std::size_t>(1, options.windowsize);
  options.windowscale = std::max<std::size_t>(1, options.windowscale);
  options.target_accept = std::isfinite(options.target_accept) ?
    std::clamp(options.target_accept, 1e-6, 1.0 - 1e-6) : default_target;

  if (!(options.min_stepsize > 0.0) ||
      !std::isfinite(options.min_stepsize)) {
    options.min_stepsize = 1e-12;
  }
  if (!(options.max_stepsize >= options.min_stepsize) ||
      !std::isfinite(options.max_stepsize)) {
    options.max_stepsize = std::max(1e3, options.min_stepsize);
  }
  if (!(options.initial_stepsize > 0.0) ||
      !std::isfinite(options.initial_stepsize)) {
    options.initial_stepsize = 1.0;
  }
  options.initial_stepsize = std::clamp(
    options.initial_stepsize, options.min_stepsize, options.max_stepsize);

  if (!(options.adam_learning_rate > 0.0) ||
      !std::isfinite(options.adam_learning_rate)) {
    options.adam_learning_rate = 0.05;
  }
  options.adam_beta1 = std::isfinite(options.adam_beta1) ?
    std::clamp(options.adam_beta1, 0.0, std::nextafter(1.0, 0.0)) : 0.9;
  options.adam_beta2 = std::isfinite(options.adam_beta2) ?
    std::clamp(options.adam_beta2, 0.0, std::nextafter(1.0, 0.0)) : 0.999;
  if (!(options.adam_epsilon > 0.0) ||
      !std::isfinite(options.adam_epsilon)) {
    options.adam_epsilon = 1e-8;
  }

  if (!(options.variance_floor > 0.0) ||
      !std::isfinite(options.variance_floor)) {
    options.variance_floor = 1e-8;
  }
  if (!(options.variance_ceiling >= options.variance_floor) ||
      !std::isfinite(options.variance_ceiling)) {
    options.variance_ceiling = std::max(1e8, options.variance_floor);
  }
  options.grad_clip = std::isfinite(options.grad_clip) ?
    std::abs(options.grad_clip) : std::numeric_limits<double>::infinity();
  return options;
}

template <typename Options>
std::size_t metric_start(const Options& options) {
  return std::min(options.initial_buffer, options.warmup);
}

template <typename Options>
std::size_t metric_end(const Options& options) {
  const std::size_t start = metric_start(options);
  if (options.warmup <= start || options.warmup <= options.terminal_buffer) {
    return start;
  }
  return std::max(start, options.warmup - options.terminal_buffer);
}

} // namespace klhr::detail
