#pragma once

#include "bfgs.hpp"
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
#include <ranges>
#include <random>
#include <string>
#include <tuple>
#include <utility>

namespace klhr {

struct KlhrOptions {
  double stepsize = 1.0;
  std::uint64_t seed = 0;
  std::size_t N = 8;
  double tol = 1e-10;
  double grad_clip = 1e15;
  double scale_clip = 300;
  double gtol = 1e-3;
  std::size_t K = 16;
  double initscale = 0.1;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t J = 2;
  double l = 0.0;
};

class KLHR {
public:

  std::size_t nfev_;
  double acceptance_rate_;

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
    acceptance_rate_ = 0.0;
    draw_ = 0;
  }

  std::size_t dim() {
    return bsm_.dim();
  }

  Eigen::VectorXd fit(const Eigen::VectorXd& rho) {
    Eigen::VectorXd mode_init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + theta_, value, grad);
      value = -value;
      g(0) = -grad.dot(rho);
    };

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init);
    nfev_ += mode.nfev;

    double log_s = 0.0;
    const double h = mode.hess_inv(0, 0);
    if (std::isfinite(h) && h > 0.0) {
      log_s = 0.5 * std::log(h) + std::log(1.25);
    }

    Eigen::VectorXd init(2);
    init << mode.x(0), log_s;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      min_kl(eta, rho, value, grad);
    };
    bfgs::BfgsResult o = bfgs::bfgs(kl, init, {.gtol = opts_.gtol, .maxiter_bfgs = 4});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  void min_kl(const Eigen::VectorXd& eta, const Eigen::VectorXd& rho,
          double& value, Eigen::VectorXd& grad) {
    auto [mu, sigma] = unpack_(eta);
    value = 0.0;
    grad = Eigen::VectorXd::Zero(2);

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
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
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

  Eigen::VectorXd fit_sas(const Eigen::VectorXd& rho) {
    Eigen::VectorXd mode_init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd target_grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + theta_, value, target_grad);
      value = -value;
      g(0) = -target_grad.dot(rho);
    };

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init);
    nfev_ += mode.nfev;

    double log_s = 0.0;
    const double h = mode.hess_inv(0, 0);
    if (std::isfinite(h) && h > 0.0) {
      log_s = 0.5 * std::log(h) + std::log(1.25);
    }

    Eigen::VectorXd init(3);
    init << mode.x(0), log_s, 0.0;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      min_sas_kl(eta, rho, value, grad);
    };
    bfgs::BfgsResult o = bfgs::bfgs(kl, init, {.gtol = opts_.gtol, .maxiter_bfgs = 4});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  void min_sas_kl(const Eigen::VectorXd& eta, const Eigen::VectorXd& rho,
                  double& value, Eigen::VectorXd& grad) {
    auto [m, s, e] = unpack_sas_(eta);
    value = 0.0;
    grad = Eigen::VectorXd::Zero(3);

    double t;
    double logp;
    double line_grad;
    Eigen::VectorXd xi(dim());
    Eigen::VectorXd grad_logp(dim());

    for (Eigen::Index n = 0; n < x_.size(); ++n) {
      const double xn = x_(n);
      const double wn = w_(n);
      const double a = std::asinh(xn) + e;
      const double sh = sinh_clipped_(a);
      const double ch = cosh_clipped_(a);
      const double th = tanh_clipped_(a);

      t = m + s * sh;
      xi = t * rho + theta_;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      line_grad = grad_logp.dot(rho);

      value += wn * (-eta(1) - log_cosh_clipped_(a) - logp);
      grad(0) -= wn * line_grad;
      grad(1) += wn * (-1.0 - line_grad * s * sh);
      grad(2) += wn * (-th - line_grad * s * ch);
    }
  }

  double log_q(const double x, const double mu, const double sigma) {
    double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  static double log_cosh(const double x) {
    const double ax = std::abs(x);
    return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
  }

  double sas_log_q(const double x, const double m, const double s, const double e) {
    const double z = (x - m) / s;
    const double y = std::asinh(z) - e;
    const double sh = sinh_clipped_(y);

    return -std::log(s) + log_cosh_clipped_(y)
      - 0.5 * sh * sh
      - 0.5 * std::log1p(z * z);
  }

  double sas_transform_d1(double normal_draw, double m, double s, double e) {
    const double z = sinh_clipped_(std::asinh(normal_draw) + e);
    return m + s * z;
  }

  double sas_to_normal_d1(double x, double m, double s, double e) {
    const double z = (x - m) / s;
    return sinh_clipped_(std::asinh(z) - e);
  }

  Eigen::VectorXd sinh_KL_step() {
    Eigen::VectorXd rho = random_direction();
    Eigen::VectorXd eta = fit_sas(rho);
    auto [mu, sigma, eps] = unpack_sas_(eta);
    double xi = overrelaxed_sas_proposal_(mu, sigma, eps);
    Eigen::VectorXd thetap = xi * rho + theta_;

    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);
    r += sas_log_q(0.0, mu, sigma, eps);
    r -= sas_log_q(xi, mu, sigma, eps);

    nfev_ += 2;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    double d = a - acceptance_rate_;
    acceptance_rate_ += d / draw_;
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd normal_KL_step() {
    Eigen::VectorXd rho = random_direction();
    Eigen::VectorXd eta = fit(rho);
    auto [mu, sigma] = unpack_(eta);
    double xi = overrelaxed_normal_proposal_(mu, sigma);
    Eigen::VectorXd thetap = xi * rho + theta_;

    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);
    r += log_q(0.0, mu, sigma);
    r -= log_q(xi, mu, sigma);

    nfev_ += 2;

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    double d = a - acceptance_rate_;
    acceptance_rate_ += d / draw_;
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
    Eigen::VectorXd theta = normal_KL_step();
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

  double overrelaxed_normal_proposal_(double mu, double sigma) {
    sigma = std::max(sigma, opts_.tol);
    const double u = normal_cdf_((0.0 - mu) / sigma);
    const double up = overrelaxed_cdf_proposal_(u);
    return mu + sigma * normal_quantile_(up);
  }

  double overrelaxed_sas_proposal_(double m, double s, double e) {
    s = std::max(s, opts_.tol);
    const double u = normal_cdf_(sas_to_normal_d1(0.0, m, s, e));
    const double up = overrelaxed_cdf_proposal_(u);
    return sas_transform_d1(normal_quantile_(up), m, s, e);
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

  static double normal_quantile_(double p) {
    p = clamp_probability_(p);

    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 =  2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 =  1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 =  2.506628277459239e+00;

    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 =  1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 =  6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;

    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 =  4.374664141464968e+00;
    constexpr double c6 =  2.938163982698783e+00;

    constexpr double d1 =  7.784695709041462e-03;
    constexpr double d2 =  3.224671290700398e-01;
    constexpr double d3 =  2.445134137142996e+00;
    constexpr double d4 =  3.754408661907416e+00;

    constexpr double plow = 0.02425;
    constexpr double phigh = 1.0 - plow;

    if (p < plow) {
      const double q = std::sqrt(-2.0 * std::log(p));
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    if (p > phigh) {
      const double q = std::sqrt(-2.0 * std::log(1.0 - p));
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    const double q = p - 0.5;
    const double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
      / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
  }

  Eigen::VectorXd normal_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

  std::pair<double, double> unpack_(const Eigen::VectorXd eta) {
    const double mu = eta(0);
    const double c = opts_.scale_clip;
    const double log_s = std::clamp(eta(1), -c, c);
    const double sigma = exp(log_s) + opts_.tol;
    return {mu, sigma};
  }

  std::tuple<double, double, double> unpack_sas_(const Eigen::VectorXd& eta) {
    const double m = eta(0);
    const double c = opts_.scale_clip;
    const double log_s = std::clamp(eta(1), -c, c);
    const double s = std::exp(log_s) + opts_.tol;
    const double e = eta(2);
    return {m, s, e};
  }

  double scale_clipped_(double x) const {
    const double c = opts_.scale_clip;
    return std::clamp(x, -c, c);
  }

  double sinh_clipped_(double x) const {
    return std::sinh(scale_clipped_(x));
  }

  double cosh_clipped_(double x) const {
    return std::cosh(scale_clipped_(x));
  }

  double tanh_clipped_(double x) const {
    return std::tanh(scale_clipped_(x));
  }

  double log_cosh_clipped_(double x) const {
    return log_cosh(scale_clipped_(x));
  }
};

} // namespace klhr
