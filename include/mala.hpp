#pragma once

#include "adam.hpp"
#include "gradient_sampler_options.hpp"

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
    opts_(detail::normalize_gradient_sampler_options(options, 0.57)),
    adam_(opts_.adam_learning_rate, opts_.adam_beta1,
          opts_.adam_beta2, opts_.adam_epsilon),
    metric_start_(detail::metric_start(opts_)),
    metric_end_(detail::metric_end(opts_)),
    metric_windowed_adaptation_(metric_end_ - metric_start_,
                                opts_.windowsize, opts_.windowscale),
    online_moments_(bsm_.dim()) {

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

    const Eigen::Index D = dim();
    variance_ = Eigen::VectorXd::Ones(D);
    grad_ = Eigen::VectorXd::Zero(D);

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    bsm_.log_density_gradient_noe(theta_, log_density_, grad_);
    ++nfev_;
    grad_ = clipped_gradient_(grad_);
    draw_ = 0;
    stepsize_ = opts_.initial_stepsize;
    log_stepsize_ = std::log(stepsize_);
    adam_.reset();
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

    const double delta = accepted - acceptance_rate_;
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
    const double target = opts_.target_accept;
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
    adam_.reset();
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
  Adam adam_;
  std::size_t metric_start_;
  std::size_t metric_end_;
  mcmcpp::WindowedAdaptation metric_windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd grad_;
  Eigen::VectorXd variance_;
  double stepsize_;
  double log_stepsize_;
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
    const Eigen::ArrayXd var = variance.array().max(opts_.variance_floor);
    const Eigen::VectorXd mean = from +
      0.5 * epsilon * epsilon *
      (var * grad_from.array()).matrix();
    const Eigen::ArrayXd diff = (to - mean).array();
    const double scale2 = epsilon * epsilon;
    const double quad = (diff.square() / (scale2 * var)).sum();
    const double logdet = to.size() * std::log(scale2) + var.log().sum();
    return -0.5 * (quad + logdet);
  }

  void adapt_warmup_(const double accept_stat) {
    if (draw_ > opts_.warmup) {
      return;
    }

    const bool in_initial = draw_ <= metric_start_;
    const bool in_metric = draw_ > metric_start_ && draw_ <= metric_end_;
    const bool in_final = draw_ > metric_end_ && draw_ <= opts_.warmup;

    if (in_metric) {
      online_moments_.update(theta_);
      const std::size_t metric_draw = draw_ - metric_start_;
      if (metric_windowed_adaptation_.window_closed(metric_draw) ||
          draw_ == metric_end_) {
        update_variance_();
        online_moments_.reset();
      }
    }

    if (in_initial || in_final) {
      if (in_final && !final_adam_reset_) {
        adam_.reset();
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
    const double gradient = accept_stat - opts_.target_accept;
    log_stepsize_ += adam_.step(gradient);
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
    return std::clamp(epsilon, opts_.min_stepsize, opts_.max_stepsize);
  }

  Eigen::VectorXd clipped_gradient_(const Eigen::VectorXd& grad) const {
    return grad.array().min(opts_.grad_clip).max(-opts_.grad_clip).matrix();
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D,
                  [&](){ return std_normal_(rng_); });
    return out;
  }

};

} // namespace klhr
