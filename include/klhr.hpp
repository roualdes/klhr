#pragma once

#include "bfgs.hpp"
#include "bridgestan.hpp"
#include "gausshermite.hpp"
#include "onlinepca.hpp"
#include "rng.hpp"
#include "sinh_asinh.hpp"
#include "welford.hpp"
#include "windowedadaptation.hpp"

#include <omp.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <random>
#include <string>
#include <tuple>
#include <utility>

namespace klhr {

struct KlhrOptions {
  double stepsize = 1.0;
  std::uint64_t seed = 0;
  std::size_t N = 16;
  double tol = 1e-10;
  double grad_clip = 1e15;
  double scale_clip = 300;
  double gtol = 1e-3;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t J = 2;
  double l = 0.0;
};

class KLHR {
public:

  std::size_t nfev_;

  KLHR(std::string stan_file, std::string json_file,
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
    draw_ = 0;
  }

  std::size_t dim() {
    return bsm_.dim();
  }

  std::pair<double, double> approx_min_kl(const Eigen::VectorXd& rho) {
    Eigen::VectorXd init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + theta_, value, grad);
      value = -value;
      g(0) = -grad.dot(rho);
    };

    bfgs::BfgsResult o = bfgs::bfgs(fg, init, {.gtol = 1e-3});
    nfev_ += o.nfev;

    double m = o.x(0);
    double h = o.hess_inv(0, 0);
    double s;
    if (std::isfinite(h) && h > 0.0) {
      s = std::sqrt(h);
    } else {
      s = approx_normal_tails_(m, 0.1 * (1 + std::abs(m)), rho);
    }
    return {m, s};
  }

  std::tuple<double, double, double> approx_sinh_kl(const Eigen::VectorXd& rho) {
    Eigen::VectorXd init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + theta_, value, grad);
      value = -value;
      g(0) = -grad.dot(rho);
    };

    bfgs::BfgsResult o = bfgs::bfgs(fg, init, {.gtol = 1e-2});
    nfev_ += o.nfev;

    double m = o.x(0);
    double q0 = -o.fun;

    double s0 = std::numeric_limits<double>::quiet_NaN();
    const double target_drop = 2.0;
    const double min_h = 1e-8;
    if (std::isfinite(o.hess_inv(0, 0)) && o.hess_inv(0, 0) > 0.0) {
      s0 = std::sqrt(o.hess_inv(0, 0));
    }
    double h = std::isfinite(s0)
      ? std::max(std::sqrt(2.0 * target_drop) * s0, min_h)
      : 0.1 * std::max(1.0, std::abs(m));

    double qp = bsm_.log_density_noe((m + h) * rho + theta_);
    double qm = bsm_.log_density_noe((m - h) * rho + theta_);
    nfev_ += 2;
    double s;
    double e;
    (void)fit_sas_scale_skew_eigen(q0, qp, qm, h, s0, s, e);

    return {m, s, e};
  }

  double fit_sas_scale_skew_eigen(double q0, double qp, double qm,
                                  double h, double s_init,
                                  double& s_hat, double& e_hat) {
    const double A_obs = qp - qm;
    const double B_obs = qp + qm - 2.0 * q0;

    double Dp = q0 - qp;
    double Dm = q0 - qm;

    const double eps = opts_.tol;
    Dp = std::max(Dp, eps);
    Dm = std::max(Dm, eps);

    double e0 = 0.25 * std::log(Dm / Dp);
    e0 = std::clamp(e0, -5.0, 5.0);

    double u0 = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(s_init) && s_init > 0.0) {
      u0 = h / s_init;
    }
    if (!std::isfinite(u0) || u0 <= 0.0) {
      u0 = std::sqrt(2.0 * std::sqrt(Dp * Dm));
    }
    u0 = std::max(u0, 1e-8);

    Eigen::VectorXd theta(2);
    theta[0] = std::log(u0);
    theta[1] = e0;

    SASFunctor functor(A_obs, B_obs);
    Eigen::NumericalDiff<SASFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SASFunctor>> lm(numDiff);

    lm.parameters.maxfev = 100;
    lm.parameters.xtol = 1e-3;
    lm.parameters.ftol = 1e-3;

    int status = lm.minimize(theta);

    const double alpha_hat = theta[0];
    const double u_hat = std::exp(alpha_hat);

    s_hat = h / u_hat;
    e_hat = theta[1];

    return static_cast<double>(status);
  }

  double approx_normal_tails_(const double m, const double x,
                              const Eigen::VectorXd& rho) {
    Eigen::VectorXd theta_m = m * rho + theta_;
    Eigen::VectorXd theta_x = x * rho + theta_;
    double dx = bsm_.log_density_noe(theta_x);
    dx -= bsm_.log_density_noe(theta_m);
    nfev_ += 2;
    double s = std::abs(x - m) / std::sqrt(2 * std::abs(dx) + opts_.tol);
    return s;
  }

  Eigen::VectorXd NURS_step() {
    Eigen::VectorXd rho = random_direction();
    auto [m, s] = approx_min_kl(rho);
    double stepsize = std_uniform_(rng_) * s / 4.0;
    double xi_step = (std_uniform_(rng_) - 0.5) * stepsize;
    double r = bsm_.log_density_noe(xi_step * rho + theta_);
    r -= bsm_.log_density_noe(theta_);
    nfev_ += 2;
    bool a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ += xi_step * rho * a;
    Eigen::Index B = 128;
    Eigen::VectorXd xis = linspace_(B, m - 40 * stepsize, m + 40 * stepsize);
    Eigen::VectorXd lds(B);

    Eigen::setNbThreads(1);
    #pragma omp parallel for num_threads(4) schedule(static)
    for (Eigen::Index b = 0; b < B; ++b) {
      lds[b] = bsm_.log_density_noe(xis(b) * rho + theta_);
    }

    nfev_ += B;
    Eigen::VectorXd probs = softmax_(lds);
    std::discrete_distribution<Eigen::Index> d(probs.data(),
                                               probs.data() + probs.size());
    double xi_prop = xis(d(rng_));
    theta_ += xi_prop * rho;
    return theta_;
  }

  Eigen::VectorXd fit(const Eigen::VectorXd& rho) {
    Eigen::VectorXd init = normal_(2);
    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      min_kl(eta, rho, value, grad);
    };
    bfgs::BfgsResult o = bfgs::bfgs(kl, init);
    return o.x;
  }

  void min_kl(const Eigen::VectorXd& eta, const Eigen::VectorXd& rho,
          double& value, Eigen::VectorXd& grad) {
    auto [mu, sigma] = unpack_(eta);
    grad.resize(2);

    auto xw =
      std::views::iota(Eigen::Index{0}, x_.size()) |
      std::views::transform([&](Eigen::Index n) {
        return std::pair<const double&, const double&>{x_(n), w_(n)};
      });

    double y;
    double logp;
    double w_grad_rho;
    Eigen::VectorXd xi(dim());
    Eigen::VectorXd grad_logp(dim());

    for (auto&& [xn, wn]: xw) {
      y = sigma * xn + mu;
      xi = y * rho + theta_;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      value += wn * logp;
      w_grad_rho = wn * grad_logp.dot(rho);
      grad(0) += w_grad_rho;
      grad(1) += w_grad_rho * xn * sigma;
    }
    value += eta(1);
    grad(1) += 1;
    value = -value;
    grad = -grad;
  }

  double log_q(const double x, const double mu, const double sigma) {
    double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  double log_cosh(const double x) {
    const double ax = std::abs(x);
    return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
  }

  double sas_log_q(const double x, const double m, const double s, const double e) {
    const double z = (x - m) / s;
    const double y = std::asinh(z) - e;
    const double sh = std::sinh(y);

    return -std::log(s) + log_cosh(y)
      - 0.5 * sh * sh
      - 0.5 * std::log1p(z * z);
  }

  double sas_transform_d1(double normal_draw, double m, double s, double e) {
    const double z = std::sinh(std::asinh(normal_draw) + e);
    return m + s * z;
  }

  double sas_to_normal_d1(double x, double m, double s, double e) {
    const double z = (x - m) / s;
    return std::sinh(std::asinh(z) - e);
  }

  Eigen::VectorXd sinh_KL_step() {
    Eigen::VectorXd rho = random_direction();
    auto [mu, sigma, eps] = approx_sinh_kl(rho);
    double xi = sas_transform_d1(std_normal_(rng_), mu, sigma, eps);
    Eigen::VectorXd thetap = xi * rho + theta_;

    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);
    r += sas_log_q(0.0, mu, sigma, eps);
    r -= sas_log_q(xi, mu, sigma, eps);

    nfev_ += 2;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd normal_KL_step() {
    Eigen::VectorXd rho = random_direction();
    // Eigen::VectorXd eta = min_kl(rho);
    // auto [mu, sigma] = unpack_(eta);
    auto [mu, sigma] = approx_min_kl(rho);
    double xi = std_normal_(rng_) * sigma + mu;
    Eigen::VectorXd thetap = xi * rho + theta_;

    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);
    r += log_q(0.0, mu, sigma);
    r -= log_q(xi, mu, sigma);

    nfev_ += 2;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd Metropolis_step() {
    Eigen::VectorXd thetap = theta_ + normal_(dim()) * opts_.stepsize;
    double r;
    double tmp;
    bsm_.log_density_noe(thetap, r);
    bsm_.log_density_noe(theta_, tmp);
    r -= tmp;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    Eigen::VectorXd theta = sinh_KL_step();
    adapt_warmup_(theta);
    return bsm_.param_constrain(theta);
  }

  Eigen::VectorXd random_direction() {
    Eigen::VectorXd weights = eigvals_.cwiseMax(0.0);
    double weight_sum = weights.sum();
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      weights.setOnes();
      weight_sum = weights.sum();
    }

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

private:

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

  void adapt_warmup_(const Eigen::VectorXd& theta) {
    if (draw_ > opts_.warmup) {
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

  Eigen::VectorXd normal_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

  std::pair<double, double> unpack_(const Eigen::VectorXd eta) {
    const double mu = eta(0);
    const double c = opts_.grad_clip;
    const double log_s = std::clamp(eta(1), -c, c);
    const double sigma = exp(log_s) + opts_.tol;
    return {mu, sigma};
  }

  Eigen::VectorXd softmax_(const Eigen::VectorXd& x) {
    const double max_coeff = x.maxCoeff();
    Eigen::VectorXd exps = (x.array() - max_coeff).exp();
    const double sum_exps = exps.sum();
    // if (!std::isfinite(sum_exps) || sum_exps == 0.0) {
    //   throw std::runtime_error("softmax failed: invalid normalization constant");
    // }
    return exps / sum_exps;
  }

  inline Eigen::VectorXd linspace_(Eigen::Index n, double lo, double hi) {
    return Eigen::VectorXd::LinSpaced(n, lo, hi);
  }
};

} // namespace klhr
