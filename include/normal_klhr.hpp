#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <utility>

namespace klhr {

class NormalKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;

  Eigen::VectorXd fit(const Eigen::VectorXd& rho) {
    return fit_at_(theta_, rho);
  }

  void min_kl(const Eigen::VectorXd& eta,
              const Eigen::VectorXd& rho,
              double& value,
              Eigen::VectorXd& grad) {
    min_kl_at_(eta, theta_, rho, value, grad);
  }

  Eigen::VectorXd normal_KL_step() {
    return regular_kl_step_();
  }

  Eigen::VectorXd normal_KL_step(const Eigen::VectorXd& rho) {
    return regular_kl_step_(rho);
  }

protected:
  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    return fit_at_(center, rho);
  }

  double overrelaxed_line_proposal_(const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    return overrelaxed_normal_proposal_(mu, sigma);
  }

  double log_line_transition_density_(double from,
                                      double to,
                                      const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    return normal_log_transition_density_(from, to, mu, sigma);
  }

  double reference_scale_(const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    (void)mu;
    return std::max(sigma, opts_.tol);
  }

  Eigen::VectorXd fit_at_(const Eigen::VectorXd& center,
                          const Eigen::VectorXd& rho) {
    Eigen::VectorXd mode_init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + center, value, grad);
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
      min_kl_at_(eta, center, rho, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol, .maxiter_bfgs = 4});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  void min_kl_at_(const Eigen::VectorXd& eta,
                  const Eigen::VectorXd& center,
                  const Eigen::VectorXd& rho,
                  double& value,
                  Eigen::VectorXd& grad) {
    auto [mu, sigma] = unpack_(eta);
    value = 0.0;
    grad = Eigen::VectorXd::Zero(2);

    double y;
    double logp;
    double w_grad_rho;
    Eigen::VectorXd xi(dim());
    Eigen::VectorXd grad_logp(dim());

    for (Eigen::Index n = 0; n < x_.size(); ++n) {
      const double xn = x_(n);
      const double wn = w_(n);
      y = sigma * xn + mu;
      xi = y * rho + center;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      value += wn * logp;
      w_grad_rho = wn * grad_logp.dot(rho);
      grad(0) += w_grad_rho;
      grad(1) += w_grad_rho * xn * sigma;
    }
    value += eta(1);
    grad(1) += 1.0;
    value = -value;
    grad = -grad;
  }

  double log_q(const double x, const double mu, const double sigma) {
    const double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  double normal_log_transition_density_(double from,
                                        double to,
                                        double mu,
                                        double sigma) {
    sigma = std::max(sigma, opts_.tol);
    const double log_density = log_q(to, mu, sigma);
    if (opts_.K == 0) {
      return log_density;
    }
    const double u_from = normal_cdf_((from - mu) / sigma);
    const double u_to = normal_cdf_((to - mu) / sigma);
    return ordered_overrelaxed_cdf_log_density_(u_from, u_to) + log_density;
  }

  double overrelaxed_normal_proposal_(double mu, double sigma) {
    sigma = std::max(sigma, opts_.tol);
    const double u = normal_cdf_((0.0 - mu) / sigma);
    const double up = overrelaxed_cdf_proposal_(u);
    return mu + sigma * normal_quantile_(clamp_probability_(up));
  }

  std::pair<double, double> unpack_(const Eigen::VectorXd& eta) {
    const double mu = eta(0);
    const double c = opts_.scale_clip;
    const double log_s = std::clamp(eta(1), -c, c);
    const double sigma = std::exp(log_s) + opts_.tol;
    return {mu, sigma};
  }
};

} // namespace klhr
