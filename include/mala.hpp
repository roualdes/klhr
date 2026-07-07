#pragma once

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
#include <string>

namespace klhr {

struct MALAOptions {
  std::uint64_t seed = 0;
  std::size_t warmup = 1'000;
  std::size_t initial_buffer = 75;
  std::size_t terminal_buffer = 50;
  std::size_t windowsize = 25;
  std::size_t windowscale = 2;
  double target_accept = 0.57;
  double initial_stepsize = 1.0;
  double min_stepsize = 1e-12;
  double max_stepsize = 1e3;
  double adam_learning_rate = 0.05;
  double adam_beta1 = 0.9;
  double adam_beta2 = 0.999;
  double adam_epsilon = 1e-8;
  double variance_floor = 1e-8;
  double variance_ceiling = 1e8;
  double grad_clip = std::numeric_limits<double>::infinity();
};

class MALA {
public:
  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;

  MALA(std::string stan_file, std::string json_file,
       const MALAOptions& options = MALAOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(options),
    metric_windowed_adaptation_(metric_warmup_(options), options.windowsize,
                                options.windowscale),
    online_moments_(bsm_.dim()) {

    if (opts_.seed == 0) {
      std::random_device rd;
      opts_.seed =
        (static_cast<std::uint64_t>(rd()) << 32) ^
        static_cast<std::uint64_t>(rd());
      if (opts_.seed == 0) {
        opts_.seed = 1;
      }
      rng_.seed(opts_.seed);
    }

    std::uniform_int_distribution<unsigned int> uniform_uint;
    mcmcpp::bsrng bsrng = bsm_.make_rng(uniform_uint(rng_));
    theta_ = bsm_.param_initialize(bsrng);

    const Eigen::Index D = dim();
    variance_ = Eigen::VectorXd::Ones(D);
    grad_ = Eigen::VectorXd::Zero(D);

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    bsm_.log_density_gradient_noe(theta_, log_density_, grad_);
    ++nfev_;
    grad_ = clipped_gradient_(grad_);
    draw_ = 0;
    stepsize_ = clamp_stepsize_(opts_.initial_stepsize);
    log_stepsize_ = std::log(stepsize_);
    reset_adam_();
    find_good_initial_stepsize(opts_.initial_stepsize);
  }

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  double stepsize() const {
    return stepsize_;
  }

  Eigen::VectorXd variance() const {
    return variance_;
  }

  Eigen::VectorXd draw() {
    ++draw_;

    MALAProposal proposal = make_proposal_(stepsize_, variance_);
    const double accept_stat = proposal.valid ?
      std::exp(std::min(0.0, proposal.log_accept)) : 0.0;
    const bool accepted =
      proposal.valid && std::log(std_uniform_(rng_)) <
      std::min(0.0, proposal.log_accept);

    const double delta = static_cast<double>(accepted) - acceptance_rate_;
    acceptance_rate_ += delta / draw_;

    if (accepted) {
      theta_ = proposal.theta;
      log_density_ = proposal.log_density;
      grad_ = proposal.grad;
    }

    adapt_warmup_(accept_stat);
    return bsm_.param_constrain(theta_);
  }

  double find_good_initial_stepsize(double initial = 1.0) {
    double epsilon = clamp_stepsize_(initial);
    const double target = target_accept_();
    double accept_stat = trial_accept_stat_(epsilon);

    if (!std::isfinite(accept_stat)) {
      accept_stat = 0.0;
    }

    if (accept_stat > target) {
      double last_good = epsilon;
      while (accept_stat > target && epsilon < opts_.max_stepsize) {
        last_good = epsilon;
        epsilon = clamp_stepsize_(2.0 * epsilon);
        if (epsilon == last_good) {
          break;
        }
        accept_stat = trial_accept_stat_(epsilon);
        if (!std::isfinite(accept_stat)) {
          accept_stat = 0.0;
        }
      }
      if (accept_stat <= target) {
        epsilon = last_good;
      }
    } else {
      while (accept_stat < target && epsilon > opts_.min_stepsize) {
        const double previous = epsilon;
        epsilon = clamp_stepsize_(0.5 * epsilon);
        if (epsilon == previous) {
          break;
        }
        accept_stat = trial_accept_stat_(epsilon);
        if (!std::isfinite(accept_stat)) {
          accept_stat = 0.0;
        }
      }
    }

    set_stepsize_(epsilon);
    reset_adam_();
    return stepsize_;
  }

private:
  struct MALAProposal {
    Eigen::VectorXd theta;
    Eigen::VectorXd grad;
    double log_density = -std::numeric_limits<double>::infinity();
    double log_accept = -std::numeric_limits<double>::infinity();
    bool valid = false;
  };

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;
  MALAOptions opts_;
  mcmcpp::WindowedAdaptation metric_windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd grad_;
  Eigen::VectorXd variance_;
  double stepsize_;
  double log_stepsize_;
  double adam_m_;
  double adam_v_;
  std::size_t adam_t_;
  std::size_t draw_;
  bool final_adam_reset_ = false;

  MALAProposal make_proposal_(const double epsilon,
                              const Eigen::VectorXd& variance) {
    MALAProposal out;
    if (!(epsilon > 0.0) || !std::isfinite(epsilon) ||
        !theta_.allFinite() || !grad_.allFinite() ||
        !variance.allFinite()) {
      return out;
    }

    Eigen::VectorXd z = normal_rng_(dim());
    Eigen::VectorXd sqrt_variance = variance.cwiseSqrt();
    Eigen::VectorXd drift =
      0.5 * epsilon * epsilon *
      (variance.array() * grad_.array()).matrix();
    out.theta = theta_ + drift +
      epsilon * (sqrt_variance.array() * z.array()).matrix();
    if (!out.theta.allFinite()) {
      return out;
    }

    out.grad = Eigen::VectorXd::Zero(dim());
    bsm_.log_density_gradient_noe(out.theta, out.log_density, out.grad);
    ++nfev_;
    if (!std::isfinite(out.log_density) || !out.grad.allFinite()) {
      return out;
    }
    out.grad = clipped_gradient_(out.grad);
    if (!out.grad.allFinite()) {
      return out;
    }

    const double forward =
      log_proposal_density_(out.theta, theta_, grad_, epsilon, variance);
    const double reverse =
      log_proposal_density_(theta_, out.theta, out.grad, epsilon, variance);
    out.log_accept = out.log_density - log_density_ + reverse - forward;
    out.valid = std::isfinite(out.log_accept);
    return out;
  }

  double trial_accept_stat_(const double epsilon) {
    MALAProposal proposal = make_proposal_(epsilon, variance_);
    if (!proposal.valid) {
      return 0.0;
    }
    return std::exp(std::min(0.0, proposal.log_accept));
  }

  double log_proposal_density_(const Eigen::VectorXd& to,
                               const Eigen::VectorXd& from,
                               const Eigen::VectorXd& grad_from,
                               const double epsilon,
                               const Eigen::VectorXd& variance) const {
    if (!(epsilon > 0.0) || !std::isfinite(epsilon) ||
        !to.allFinite() || !from.allFinite() || !grad_from.allFinite() ||
        !variance.allFinite()) {
      return -std::numeric_limits<double>::infinity();
    }

    const Eigen::ArrayXd var = variance.array().max(opts_.variance_floor);
    const Eigen::VectorXd mean = from +
      0.5 * epsilon * epsilon *
      (var * grad_from.array()).matrix();
    const Eigen::ArrayXd diff = (to - mean).array();
    const double scale2 = epsilon * epsilon;
    const double quad = (diff.square() / (scale2 * var)).sum();
    const double logdet =
      static_cast<double>(to.size()) * std::log(scale2) + var.log().sum();
    return -0.5 * (quad + logdet);
  }

  void adapt_warmup_(const double accept_stat) {
    if (draw_ == 0 || draw_ > opts_.warmup) {
      return;
    }

    const std::size_t start = metric_start_();
    const std::size_t end = metric_end_();
    const bool in_initial = draw_ <= start;
    const bool in_metric = draw_ > start && draw_ <= end;
    const bool in_final = draw_ > end && draw_ <= opts_.warmup;

    if (in_metric) {
      online_moments_.update(theta_);
      const std::size_t metric_draw = draw_ - start;
      if (metric_windowed_adaptation_.window_closed(metric_draw) ||
          draw_ == end) {
        update_variance_();
        online_moments_.reset();
      }
    }

    if (in_initial || in_final) {
      if (in_final && !final_adam_reset_) {
        reset_adam_();
        final_adam_reset_ = true;
      }
      update_stepsize_(accept_stat);
    }
  }

  void update_variance_() {
    Eigen::VectorXd v = online_moments_.variance();
    for (Eigen::Index d = 0; d < v.size(); ++d) {
      if (!std::isfinite(v(d))) {
        v(d) = 1.0;
      }
      v(d) = std::clamp(v(d), opts_.variance_floor, opts_.variance_ceiling);
    }
    variance_ = v;
  }

  void update_stepsize_(const double accept_stat) {
    const double gradient = accept_stat - target_accept_();
    ++adam_t_;
    adam_m_ = opts_.adam_beta1 * adam_m_ +
      (1.0 - opts_.adam_beta1) * gradient;
    adam_v_ = opts_.adam_beta2 * adam_v_ +
      (1.0 - opts_.adam_beta2) * gradient * gradient;
    const double m_hat =
      adam_m_ / (1.0 - std::pow(opts_.adam_beta1, adam_t_));
    const double v_hat =
      adam_v_ / (1.0 - std::pow(opts_.adam_beta2, adam_t_));
    log_stepsize_ += opts_.adam_learning_rate * m_hat /
      (std::sqrt(v_hat) + opts_.adam_epsilon);
    set_stepsize_(std::exp(log_stepsize_));
  }

  void set_stepsize_(const double epsilon) {
    stepsize_ = clamp_stepsize_(epsilon);
    log_stepsize_ = std::log(stepsize_);
  }

  double clamp_stepsize_(double epsilon) const {
    if (!std::isfinite(epsilon) || !(epsilon > 0.0)) {
      epsilon = 1.0;
    }
    const double lo = std::max(opts_.min_stepsize,
                               std::numeric_limits<double>::min());
    const double hi = std::max(opts_.max_stepsize, lo);
    return std::clamp(epsilon, lo, hi);
  }

  double target_accept_() const {
    return std::clamp(opts_.target_accept, 1e-6, 1.0 - 1e-6);
  }

  void reset_adam_() {
    adam_m_ = 0.0;
    adam_v_ = 0.0;
    adam_t_ = 0;
  }

  Eigen::VectorXd clipped_gradient_(const Eigen::VectorXd& grad) const {
    if (!std::isfinite(opts_.grad_clip)) {
      return grad;
    }
    return grad.array().min(opts_.grad_clip).max(-opts_.grad_clip).matrix();
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D,
                  [&](){ return std_normal_(rng_); });
    return out;
  }

  std::size_t metric_start_() const {
    return std::min(opts_.initial_buffer, opts_.warmup);
  }

  std::size_t metric_end_() const {
    const std::size_t start = metric_start_();
    if (opts_.warmup <= start || opts_.warmup <= opts_.terminal_buffer) {
      return start;
    }
    return std::max(start, opts_.warmup - opts_.terminal_buffer);
  }

  static std::size_t metric_warmup_(const MALAOptions& options) {
    const std::size_t start =
      std::min(options.initial_buffer, options.warmup);
    if (options.warmup <= start || options.warmup <= options.terminal_buffer) {
      return 0;
    }
    return std::max(start, options.warmup - options.terminal_buffer) - start;
  }
};

} // namespace klhr
