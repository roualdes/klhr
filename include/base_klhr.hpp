#pragma once

#include "bridgestan.hpp"
#include "gausshermite.hpp"

#include "onlinepca.hpp"
#include "rng.hpp"
#include "welford.hpp"
#include "windowedadaptation.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <string>

#include <iostream>

namespace klhr {

struct KlhrOptions {
  double stepsize = 1.0;
  std::uint64_t seed = 0;
  std::size_t N = 8;
  double tol = 1e-10;
  double grad_clip = std::numeric_limits<double>::infinity(); // 1e15; //
  double scale_clip = 300;
  double gtol = 1e-3;
  std::size_t K = 16;
  double initscale = 0.1;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t J = 4;
  double l = 0.0;
  double initial_transport_gradient_floor = 1e-8;
};

class BaseKLHR {
public:

  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;

  BaseKLHR(std::string stan_file, std::string json_file,
           const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    uniform_uint_(),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(options),
    windowed_adaptation_(options.warmup, options.windowsize, options.windowscale),
    online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), options.J, options.l, options.tol) {

    if (opts_.seed == 0) {
      std::random_device rd;
      rng_.seed(rd());
    }

    bsrng_ = bsm_.make_rng(uniform_uint_(rng_));
    theta_.resize(dim());
    theta_ = bsm_.param_initialize(bsrng_);

    w_.resize(opts_.N);
    x_.resize(opts_.N);
    gauss_hermite(opts_.N, w_, x_);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);

    const Eigen::Index D = static_cast<Eigen::Index>(dim());
    mean_ = Eigen::VectorXd::Zero(D);
    cov_ = Eigen::VectorXd::Ones(D);
    eigvecs_ = Eigen::MatrixXd::Zero(D, static_cast<Eigen::Index>(opts_.J + 1));
    eigvals_ = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(opts_.J + 1));

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    log_density_ = bsm_.log_density_noe(theta_);
    ++nfev_;
    draw_ = 0;
    warmup_ = opts_.warmup;
  }

  virtual ~BaseKLHR() = default;

  std::size_t dim() {
    return bsm_.dim();
  }

  static double log_sum_exp_2_(double a, double b) {
    if (!std::isfinite(a)) {
      return b;
    }
    if (!std::isfinite(b)) {
      return a;
    }
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
  }

  Eigen::VectorXd KL_step() {
    return regular_kl_step_();
  }

  Eigen::VectorXd KL_step(const Eigen::VectorXd& rho) {
    return regular_kl_step_(rho);
  }

  Eigen::VectorXd Metropolis_step() {
    Eigen::VectorXd thetap = theta_ + normal_(dim()) * opts_.stepsize;
    double proposal_logp;
    bsm_.log_density_noe(thetap, proposal_logp);
    ++nfev_;
    double r = proposal_logp - log_density_;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    if (a) {
      theta_ = thetap;
      log_density_ = proposal_logp;
    }
    return theta_;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    Eigen::VectorXd theta;
    theta = regular_kl_step_();
    adapt_warmup_(theta);
    return bsm_.param_constrain(theta);
  }

  Eigen::VectorXd random_direction() {
    Eigen::VectorXd weights = eigvals_.cwiseMax(0.0);
    std::discrete_distribution<std::size_t> component(weights.data(),
                                                      weights.data() + weights.size());
    const std::size_t j = component(rng_);

    Eigen::VectorXd rho(dim());
    const Eigen::VectorXd center = eigvecs_.col(static_cast<Eigen::Index>(j));
    for (Eigen::Index d = 0; d < rho.size(); ++d) {
      const double variance = std::max(cov_(d), opts_.tol);
      rho(d) = center(d) + std::sqrt(variance) * std_normal_(rng_);
    }

    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      rho = normal_(dim());
      norm = rho.norm();
    }
    return rho / (norm + opts_.tol);
  }

protected:

  struct StepStats {
    double accept_prob = 0.0;
    double scaled_jump = 0.0;
    bool accepted = false;
  };

  virtual Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                                    const Eigen::VectorXd& rho) = 0;

  virtual double overrelaxed_line_proposal_(const Eigen::VectorXd& eta) = 0;

  virtual double log_line_transition_density_(double from,
                                              double to,
                                              const Eigen::VectorXd& eta) = 0;

  virtual double reference_scale_(const Eigen::VectorXd& eta) = 0;

  mcmcpp::bsmodel bsm_;
  mcmcpp::bsrng bsrng_;
  mcmcpp::rng rng_;

  std::uniform_int_distribution<unsigned int> uniform_uint_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  WindowedAdaptation windowed_adaptation_;
  WelfordAccumulator online_moments_;
  OnlinePCA online_pca_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Guass-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;

  std::size_t draw_;
  std::size_t warmup_;

  Eigen::VectorXd regular_kl_step_() {
    return regular_kl_step_(random_direction());
  }

  Eigen::VectorXd regular_kl_step_(const Eigen::VectorXd& rho) {
    const Eigen::VectorXd theta_before = theta_;
    Eigen::VectorXd grad(dim());
    double current_logp = log_density_;
    bsm_.log_density_gradient_noe(theta_, current_logp, grad);
    ++nfev_;
    if (std::isfinite(current_logp)) {
      log_density_ = current_logp;
    } else {
      current_logp = log_density_;
    }
    sanitize_gradient_(grad);

    Eigen::VectorXd eta = fit_line_(theta_before, rho);
    const double xi = overrelaxed_line_proposal_(eta);

    Eigen::VectorXd thetap = theta_;
    double proposal_logp = log_density_;
    double r = 0.0;
    if (!std::isfinite(xi)) {
      proposal_logp = -std::numeric_limits<double>::infinity();
      r = -std::numeric_limits<double>::infinity();
    } else if (std::abs(xi) > opts_.tol) {
      thetap = xi * rho + theta_;
      proposal_logp = -std::numeric_limits<double>::infinity();
      r = -std::numeric_limits<double>::infinity();
    }
    if (std::isfinite(xi) && std::abs(xi) > opts_.tol &&
        thetap.allFinite()) {
      proposal_logp = bsm_.log_density_noe(thetap);
      ++nfev_;
      if (std::isfinite(proposal_logp)) {
        const double forward_transition =
          log_line_transition_density_(0.0, xi, eta);
        Eigen::VectorXd reverse_eta = fit_line_(thetap, rho);
        const double reverse_transition =
          log_line_transition_density_(0.0, -xi, reverse_eta);
        if (std::isfinite(forward_transition) &&
            std::isfinite(reverse_transition)) {
          r = proposal_logp - log_density_;
          r += reverse_transition;
          r -= forward_transition;
        }
      }
    }

    const bool accepted = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    const double d = static_cast<double>(accepted) - acceptance_rate_;
    acceptance_rate_ += d / draw_;
    if (accepted) {
      theta_ = thetap;
      log_density_ = proposal_logp;
    }
    return theta_;
  }

  void sanitize_gradient_(Eigen::VectorXd& grad) const {
    for (Eigen::Index d = 0; d < grad.size(); ++d) {
      if (!std::isfinite(grad(d))) {
        grad(d) = 0.0;
      } else {
        grad(d) = std::clamp(grad(d), -opts_.grad_clip, opts_.grad_clip);
      }
    }
  }

  void adapt_warmup_(const Eigen::VectorXd& theta) {
    if (draw_ > warmup_) {
      return;
    }

    if (windowed_adaptation_.window_closed(draw_)) {
      mean_ = online_moments_.mean();
      cov_ = online_moments_.variance();
      online_moments_.reset();

      if (opts_.J > 0) {
        eigvecs_.leftCols(static_cast<Eigen::Index>(opts_.J)) = online_pca_.vectors();
        eigvals_.head(static_cast<Eigen::Index>(opts_.J)) = online_pca_.values();
      }
      online_pca_.reset();
    } else {
      online_moments_.update(theta);
      online_pca_.update(theta - mean_);
    }
  }

  double overrelaxed_cdf_proposal_(double u) {
    u = clamp_probability_(u);
    if (opts_.K == 0) {
      return clamp_probability_(std_uniform_(rng_));
    }

    const int K = static_cast<int>(std::min<std::size_t>(
      opts_.K, static_cast<std::size_t>(std::numeric_limits<int>::max())));

    std::binomial_distribution<int> binomial(K, u);
    const int r = binomial(rng_);

    double up = u;
    if (r > K - r) {
      const double v = beta_(static_cast<double>(K - r + 1),
                             static_cast<double>(2 * r - K));
      up = u * v;
    } else if (r < K - r) {
      const double v = beta_(static_cast<double>(r + 1),
                             static_cast<double>(K - 2 * r));
      up = 1.0 - (1.0 - u) * v;
    }

    return clamp_probability_(up);
  }

  double ordered_overrelaxed_cdf_log_density_(double from, double to) const {
    from = clamp_probability_(from);
    to = clamp_probability_(to);
    if (opts_.K == 0) {
      return 0.0;
    }
    if (from == to) {
      return -std::numeric_limits<double>::infinity();
    }

    const int K = static_cast<int>(std::min<std::size_t>(
      opts_.K, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    double log_density = -std::numeric_limits<double>::infinity();
    if (to < from) {
      const double log_from = std::log(from);
      const double v = to / from;
      for (int r = K / 2 + 1; r <= K; ++r) {
        const double a = static_cast<double>(K - r + 1);
        const double b = static_cast<double>(2 * r - K);
        const double term =
          log_binomial_pmf_(K, r, from) + log_beta_pdf_(v, a, b) -
          log_from;
        log_density = log_sum_exp_2_(log_density, term);
      }
    } else {
      const double log_one_minus_from = std::log1p(-from);
      const double v = (1.0 - to) / (1.0 - from);
      for (int r = 0; r < (K + 1) / 2; ++r) {
        const double a = static_cast<double>(r + 1);
        const double b = static_cast<double>(K - 2 * r);
        const double term =
          log_binomial_pmf_(K, r, from) + log_beta_pdf_(v, a, b) -
          log_one_minus_from;
        log_density = log_sum_exp_2_(log_density, term);
      }
    }
    return log_density;
  }

  static double log_binomial_pmf_(int n, int k, double p) {
    p = clamp_probability_(p);
    return std::lgamma(static_cast<double>(n) + 1.0) -
      std::lgamma(static_cast<double>(k) + 1.0) -
      std::lgamma(static_cast<double>(n - k) + 1.0) +
      static_cast<double>(k) * std::log(p) +
      static_cast<double>(n - k) * std::log1p(-p);
  }

  static double log_beta_pdf_(double x, double a, double b) {
    if (!(x > 0.0 && x < 1.0) || !(a > 0.0 && b > 0.0)) {
      return -std::numeric_limits<double>::infinity();
    }
    return (a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) +
      std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
  }

  double beta_(double a, double b) {
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);

    double x = gamma_a(rng_);
    double y = gamma_b(rng_);
    double total = x + y;
    while (!std::isfinite(total) || total <= 0.0) {
      x = gamma_a(rng_);
      y = gamma_b(rng_);
      total = x + y;
    }
    return x / total;
  }

  static double clamp_probability_(double p) {
    const double eps = std::numeric_limits<double>::epsilon();
    if (!std::isfinite(p)) {
      return 0.5;
    }
    return std::clamp(p, eps, 1.0 - eps);
  }

  static double normal_cdf_(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
  }

  Eigen::VectorXd normal_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

};

} // namespace klhr
