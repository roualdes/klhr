#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <utility>

namespace klhr {

class NormalKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;
protected:
    Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                          const Eigen::VectorXd& rho) override {
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
      log_s = 0.5 * std::log(h * 1.1);
    }

    Eigen::VectorXd init(2);
    init << mode.x(0), log_s;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      KL_(eta, center, rho, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 4});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  double overrelaxed_proposal_(const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    return overrelaxed_normal_proposal_(mu, sigma);
  }

  double transition_density_(const double from, const double to,
                             const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    return normal_transition_density_(from, to, mu, sigma);
  }

  void KL_(const Eigen::VectorXd& eta, const Eigen::VectorXd& center,
                  const Eigen::VectorXd& rho, double& value, Eigen::VectorXd& grad) {
    auto [mu, sigma] = unpack_(eta);
    value = 0.0;
    grad = Eigen::VectorXd::Zero(2);

    double y;
    double logp;
    double w_grad_rho;
    Eigen::Index D = dim();
    Eigen::VectorXd xi(D);
    Eigen::VectorXd grad_logp(D);

    // TODO openmp it up, try changing N
    for (Eigen::Index n = 0; n < opts_.N; ++n) {
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

  double log_q_(const double x, const double mu, const double sigma) {
    const double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  double normal_transition_density_(const double from, const double to,
                                    const double mu, const double sigma) {
    const double s = std::max(sigma, opts_.tol);
    const double log_density = log_q_(to, mu, s);
    if (opts_.K == 0) {
      return log_density;
    }
    const double u_from = normal_cdf_((from - mu) / s);
    const double u_to = normal_cdf_((to - mu) / s);
    return overrelaxed_density_(u_from, u_to) + log_density;
  }

  double overrelaxed_normal_proposal_(const double mu, const double sigma) {
    const double s = std::max(sigma, opts_.tol);
    const double u = normal_cdf_((0.0 - mu) / s);
    const double up = overrelaxed_proposal_impl_(u);
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
