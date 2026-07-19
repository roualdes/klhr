#pragma once

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
  Eigen::Index J = 1;
  double direction_lowrank_weight = 1.0;
  double direction_min_diag_fraction = 0.1;
  bool lowrank_during_warmup = true;
  double pca_freeze_fraction = 0.1;
  double l = 0.0;
  double tol = 1e-10;
  double initial_width = 1.0;
  double min_width = 1e-8;
  double max_width = 1e8;
  std::size_t max_steps_out = 100;
  std::size_t max_shrink_steps = 1'000;
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
    opts_(normalized_options_(options, bsm_.dim())),
    windowed_adaptation_(opts_.warmup, opts_.windowsize, opts_.windowscale),
    online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), opts_.J, opts_.l, opts_.tol),
    projected_moments_(opts_.J) {

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

    initialize_pca_schedule_();
    reset_adaptation_();
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

  Eigen::VectorXd variance() const {
    return diagonal_variance_();
  }

  Eigen::VectorXd draw() {
    ++draw_;
    const Eigen::VectorXd rho = random_direction_();
    last_width_ = line_width_(rho);
    const bool success = slice_step_(rho, last_width_);
    const double delta = success - acceptance_rate_;
    acceptance_rate_ += delta / draw_;
    adapt_warmup_(theta_, draw_);
    return bsm_.param_constrain(theta_);
  }

private:
  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;
  SliceOptions opts_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  OnlinePCA online_pca_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
  Eigen::MatrixXd projection_basis_;
  Eigen::MatrixXd mean_direction_basis_;
  Eigen::VectorXd mean_direction_weights_;
  bool projection_basis_ready_ = false;
  bool mean_direction_ready_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  bool pca_calibration_enabled_ = false;
  std::size_t pca_freeze_draw_ = 0;
  std::size_t projected_pair_count_ = 0;
  std::size_t draw_ = 0;
  double last_width_ = 1.0;

  static SliceOptions normalized_options_(SliceOptions options,
                                          const Eigen::Index dim) {
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    options.J = std::clamp(options.J, Eigen::Index{0}, dim);
    options.direction_lowrank_weight =
      std::isfinite(options.direction_lowrank_weight) ?
      std::clamp(options.direction_lowrank_weight, 0.0, 1.0) : 1.0;
    options.direction_min_diag_fraction =
      std::isfinite(options.direction_min_diag_fraction) ?
      std::clamp(options.direction_min_diag_fraction, 0.0, 1.0) : 0.1;
    options.pca_freeze_fraction =
      std::isfinite(options.pca_freeze_fraction) ?
      std::clamp(options.pca_freeze_fraction, 0.0, 1.0) : 0.1;
    if (!std::isfinite(options.l)) {
      options.l = 0.0;
    }
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
    options.max_steps_out = std::max<std::size_t>(1, options.max_steps_out);
    options.max_shrink_steps =
      std::max<std::size_t>(1, options.max_shrink_steps);
    return options;
  }

  bool slice_step_(const Eigen::VectorXd& rho, const double width) {
    const double u_slice = std::max(
      std_uniform_(rng_), std::numeric_limits<double>::min());
    const double log_slice = log_density_ + std::log(u_slice);

    double left = -std_uniform_(rng_) * width;
    double right = left + width;
    std::uniform_int_distribution<std::size_t> left_budget(
      0, opts_.max_steps_out - 1);
    std::size_t steps_left = left_budget(rng_);
    std::size_t steps_right = opts_.max_steps_out - 1 - steps_left;

    while (steps_left > 0 &&
           line_log_density_(left, rho) > log_slice) {
      left -= width;
      --steps_left;
      if (!std::isfinite(left)) {
        return false;
      }
    }
    while (steps_right > 0 &&
           line_log_density_(right, rho) > log_slice) {
      right += width;
      --steps_right;
      if (!std::isfinite(right)) {
        return false;
      }
    }

    for (std::size_t shrink = 0;
         shrink < opts_.max_shrink_steps; ++shrink) {
      if (!(left < right) || !std::isfinite(left) ||
          !std::isfinite(right)) {
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
      (rho.array().square() * diagonal_variance_().array()).sum();
    if (!(projected_variance > 0.0) ||
        !std::isfinite(projected_variance)) {
      return opts_.initial_width;
    }
    const double width = opts_.initial_width *
      std::sqrt(projected_variance);
    return std::clamp(width, opts_.min_width, opts_.max_width);
  }

  Eigen::VectorXd random_direction_() {
    const bool use_sampling_direction =
      (opts_.lowrank_during_warmup || draw_ > opts_.warmup) &&
      lowrank_ready_;
    Eigen::VectorXd rho = use_sampling_direction ?
      direction_noise_() : mean_direction_noise_();
    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_rng_(dim());
      norm = rho.norm();
    }
    rho /= norm + opts_.tol;
    return rho;
  }

  void reset_adaptation_() {
    const Eigen::Index D = dim();
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    mean_direction_weights_ = Eigen::VectorXd::Ones(opts_.J);
    online_moments_.reset();
    online_pca_.reset();
    projected_moments_.reset();
    windowed_adaptation_.reset();
    projection_basis_ready_ = false;
    mean_direction_ready_ = false;
    pca_frozen_ = false;
    lowrank_ready_ = false;
    projected_pair_count_ = 0;
  }

  Eigen::VectorXd diagonal_variance_() const {
    Eigen::VectorXd variance = cov_;
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (!std::isfinite(variance(d)) || variance(d) <= 0.0) {
        variance(d) = 1.0;
      } else {
        variance(d) = std::max(variance(d), opts_.tol);
      }
    }
    return variance;
  }

  Eigen::VectorXd direction_noise_() {
    const Eigen::Index D = dim();
    const double alpha = opts_.direction_lowrank_weight;
    const double min_diag_fraction = opts_.direction_min_diag_fraction;
    const Eigen::VectorXd base_variance = diagonal_variance_();
    Eigen::VectorXd residual_variance = base_variance;
    const Eigen::Index rank = lowrank_ready_ ? opts_.J : 0;
    if (rank == 0 || alpha == 0.0) {
      return diagonal_direction_noise_(base_variance);
    }

    const Eigen::MatrixXd lowrank_basis = eigvecs_.leftCols(rank);
    const Eigen::VectorXd lowrank_variance = eigvals_.head(rank);
    residual_variance -= alpha *
      (lowrank_basis.array().square().matrix() * lowrank_variance);
    for (Eigen::Index d = 0; d < D; ++d) {
      const double floor = std::max(
        opts_.tol, min_diag_fraction * base_variance(d));
      if (!std::isfinite(residual_variance(d)) ||
          residual_variance(d) < floor) {
        residual_variance(d) = floor;
      }
    }

    Eigen::VectorXd noise = diagonal_direction_noise_(residual_variance);
    const Eigen::VectorXd lowrank_sd =
      (alpha * lowrank_variance.array()).sqrt().matrix();
    noise += lowrank_basis *
      lowrank_sd.cwiseProduct(normal_rng_(rank));
    return noise.allFinite() ? noise : normal_rng_(D);
  }

  Eigen::VectorXd diagonal_direction_noise_() {
    return diagonal_direction_noise_(diagonal_variance_());
  }

  Eigen::VectorXd diagonal_direction_noise_(
      const Eigen::VectorXd& variance) {
    return variance.array().sqrt().matrix().cwiseProduct(
      normal_rng_(dim()));
  }

  Eigen::VectorXd mean_direction_noise_() {
    if (!mean_direction_ready_) {
      return diagonal_direction_noise_();
    }
    std::discrete_distribution<Eigen::Index> component(
      mean_direction_weights_.data(),
      mean_direction_weights_.data() + mean_direction_weights_.size());
    const Eigen::Index j = component(rng_);
    return diagonal_direction_noise_() + mean_direction_basis_.col(j);
  }

  void initialize_pca_schedule_() {
    std::size_t final_start = 0;
    const auto& closures = windowed_adaptation_.closures();
    if (opts_.warmup > 0) {
      final_start = 1;
      if (closures.size() >= 2 && closures.back() == opts_.warmup) {
        final_start = closures[closures.size() - 2] + 1;
      }
    }

    const std::size_t final_length = final_start <= opts_.warmup ?
      opts_.warmup - final_start + 1 : 0;
    pca_calibration_enabled_ =
      opts_.J > 0 && opts_.pca_freeze_fraction > 0.0 && final_length > 2;
    if (!pca_calibration_enabled_) {
      pca_freeze_draw_ = opts_.warmup;
      return;
    }

    const auto tail = std::size_t(
      std::ceil(opts_.pca_freeze_fraction * final_length));
    const std::size_t tail_length =
      std::clamp<std::size_t>(tail, 1, final_length);
    pca_freeze_draw_ =
      std::max(final_start, opts_.warmup - tail_length);
  }

  bool set_projection_basis_from_online_pca_() {
    if (opts_.J <= 0 || online_pca_.count() < opts_.J) {
      return false;
    }
    Eigen::MatrixXd basis = online_pca_.vectors();
    if (!basis.allFinite()) {
      return false;
    }

    projection_basis_ = basis.leftCols(opts_.J);
    for (Eigen::Index j = 0; j < projection_basis_.cols(); ++j) {
      const double norm = projection_basis_.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        projection_basis_ready_ = false;
        return false;
      }
      projection_basis_.col(j) /= norm;
    }
    projection_basis_ready_ = true;
    return true;
  }

  bool set_mean_direction_(Eigen::MatrixXd basis,
                           Eigen::VectorXd weights) {
    mean_direction_ready_ = false;
    if (!basis.allFinite() || !weights.allFinite()) {
      mean_direction_basis_.setZero();
      mean_direction_weights_.setOnes();
      return false;
    }
    for (Eigen::Index j = 0; j < basis.cols(); ++j) {
      const double norm = basis.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        mean_direction_basis_.setZero();
        mean_direction_weights_.setOnes();
        return false;
      }
      basis.col(j) /= norm;
    }
    mean_direction_basis_ = std::move(basis);
    mean_direction_weights_ = weights.cwiseMax(opts_.tol);
    mean_direction_ready_ = true;
    return true;
  }

  void set_mean_direction_from_online_pca_() {
    if (opts_.J <= 0 || online_pca_.count() < opts_.J) {
      return;
    }
    Eigen::MatrixXd basis = online_pca_.vectors().leftCols(opts_.J);
    Eigen::VectorXd weights = online_pca_.values().head(opts_.J);
    (void) set_mean_direction_(std::move(basis), std::move(weights));
  }

  void update_projected_moments_(const Eigen::VectorXd& theta) {
    if (!projection_basis_ready_) {
      return;
    }
    const Eigen::VectorXd projected = projection_basis_.transpose() * theta;
    if (projected.allFinite()) {
      projected_moments_.update(projected);
    }
  }

  void activate_projected_pair_() {
    if (!projection_basis_ready_ || projected_moments_.count() <= 2) {
      return;
    }
    Eigen::VectorXd variances = projected_moments_.variance();
    for (Eigen::Index j = 0; j < opts_.J; ++j) {
      if (!std::isfinite(variances(j)) || variances(j) <= opts_.tol) {
        variances(j) = opts_.tol;
      }
    }
    eigvecs_.leftCols(opts_.J) = projection_basis_;
    eigvals_.head(opts_.J) = variances;
    ++projected_pair_count_;
    lowrank_ready_ = projected_pair_count_ >= 2;
  }

  void freeze_pca_for_final_calibration_() {
    if (pca_frozen_) {
      return;
    }
    const bool frozen = set_projection_basis_from_online_pca_();
    set_mean_direction_from_online_pca_();
    if (!frozen && projected_pair_count_ > 0) {
      projection_basis_ = eigvecs_.leftCols(opts_.J);
      projection_basis_ready_ = projection_basis_.allFinite();
    }
    projected_moments_.reset();
    online_pca_.reset();
    pca_frozen_ = true;
  }

  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    if (adaptation_draw > opts_.warmup) {
      return;
    }
    if (pca_calibration_enabled_ && !pca_frozen_ &&
        adaptation_draw >= pca_freeze_draw_) {
      freeze_pca_for_final_calibration_();
    }

    online_moments_.update(theta);
    if (pca_calibration_enabled_) {
      update_projected_moments_(theta);
      if (!pca_frozen_) {
        online_pca_.update(theta - mean_);
      }
    }

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();
      if (pca_calibration_enabled_ && !pca_frozen_) {
        activate_projected_pair_();
        const bool has_next_basis = set_projection_basis_from_online_pca_();
        set_mean_direction_from_online_pca_();
        projected_moments_.reset();
        if (!has_next_basis) {
          projection_basis_ready_ = false;
        }
        online_pca_.reset();
      }
    }

    if (pca_calibration_enabled_ && pca_frozen_ &&
        adaptation_draw == opts_.warmup) {
      activate_projected_pair_();
    }
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index size) {
    Eigen::VectorXd out(size);
    std::generate(out.data(), out.data() + size,
                  [&]() { return std_normal_(rng_); });
    return out;
  }
};

} // namespace klhr
