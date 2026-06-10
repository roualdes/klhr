#pragma once

#include "bfgs.hpp"
#include "bridgestan.hpp"
#include "gausshermite.hpp"
#include "rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>
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
};

class KLHR {
public:

  KLHR(std::string stan_file, std::string json_file,
       const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    uniform_uint_(),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(options) {

    if (opts_.seed == 0) {
      std::random_device rd;
      rng_.seed(rd());
    }

    bsrng_ = bsm_.make_rng(uniform_uint_(rng_));
    theta_.resize(dim());
    theta_ = bsm_.param_initialize(bsrng_);

    std::tie(x_, w_) = gauss_hermite(opts_.N);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);
  }

  double log_density() {
    return bsm_.log_density_noe(theta_);
  }

  std::size_t dim() {
    return bsm_.dim();
  }

  Eigen::VectorXd minimize() {
    auto fg = [&](const Eigen::Ref<const Eigen::VectorXd>& x) -> std::pair<double, Eigen::VectorXd> {
      auto [ld, grad] = bsm_.log_density_gradient_noe(x);
      return {-ld, -grad};
    };

    bfgs::BfgsResult o = bfgs::bfgs(fg, theta_);
    return bsm_.param_constrain(o.x);
  }

  std::pair<double, Eigen::VectorXd>
  KL(const Eigen::VectorXd& eta, const Eigen::VectorXd& rho) {
    auto [mu, sigma] = unpack_(eta);
    double out = 0.0;
    Eigen::VectorXd grad(2);

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
      xi = y * rho.array() + theta_.array();
      std::tie(logp, grad_logp) = bsm_.log_density_gradient_noe(xi);
      out += wn * logp;
      w_grad_rho = wn * grad_logp.dot(rho);
      grad(0) += w_grad_rho;
      grad(1) += w_grad_rho * xn * sigma;
    }
    out += eta(1);
    grad(1) += 1;
    return {-out, -grad};
  }

  Eigen::VectorXd fit(const Eigen::VectorXd& rho) {
    Eigen::VectorXd init = normal_(2);
    auto kl = [&, this](const Eigen::VectorXd& eta) {
      return KL(eta, rho);
    };
    bfgs::BfgsResult o = bfgs::bfgs(kl, init);
    return o.x;
  }

  double log_q(const double x, const double mu, const double sigma) {
    double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  Eigen::VectorXd KL_step() {
    Eigen::VectorXd rho = random_direction_();
    Eigen::VectorXd eta = fit(rho);
    auto [mu, sigma] = unpack_(eta);
    double xi = std_normal_(rng_) * sigma + mu;
    Eigen::VectorXd thetap = xi * rho.array() + theta_.array();

    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);
    r += log_q(0.0, mu, sigma);
    r -= log_q(xi, mu, sigma);

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd Metropolis_step() {
    Eigen::VectorXd thetap = theta_ + normal_(dim()) * opts_.stepsize;
    double r = bsm_.log_density_noe(thetap);
    r -= bsm_.log_density_noe(theta_);

    double a = std::log(std_uniform_(rng_)) < std::min(0.0, r);
    theta_ = a * thetap + (1 - a) * theta_;
    return theta_;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    Eigen::VectorXd theta = Metropolis_step();
    return theta;
  }


private:

  mcmcpp::bsmodel bsm_;
  mcmcpp::bsrng bsrng_;
  mcmcpp::rng rng_;

  std::uniform_int_distribution<unsigned int> uniform_uint_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Guass-Hermite sample points
  Eigen::VectorXd w_; // and weights
  std::size_t draw_;

  Eigen::VectorXd normal_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

  Eigen::VectorXd random_direction_() {
    return normal_(dim());
  }

  double clip_(const double s) const {
    double out = s;
    if (s < -opts_.grad_clip) {
      out = -opts_.grad_clip;
    } else if (s > opts_.grad_clip){
      out = opts_.grad_clip;
    }
    return out;
  }

  std::pair<double, double> unpack_(const Eigen::VectorXd eta) {
    const double mu = eta(0);
    const double sigma = exp(clip_(eta(1))) + opts_.tol;
    return {mu, sigma};
  }


};

} // namespace klhr
