#pragma once

#include "base_klhr.hpp"
#include "normal_quantile.hpp"

#include <utility>

namespace klhr {

class NormalKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;

protected:
  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    return fit_line_with_kl_fallback_(
      center, rho, 2,
      [this, &center, &rho](const Eigen::VectorXd& eta,
                            const double log_scale,
                            double& value, Eigen::VectorXd& grad) {
        KL_(eta, center, rho, log_scale, value, grad);
      },
      [this](const Eigen::VectorXd& raw, const double log_scale) {
        Eigen::VectorXd eta(2);
        eta << raw(0), relative_log_scale_(raw(1), log_scale);
        return eta;
      });
  }

  double overrelaxed_proposal_(const Eigen::VectorXd& eta) override {
    auto [mu, sigma] = unpack_(eta);
    return overrelaxed_normal_proposal_(mu, sigma);
  }

  double transition_density_(const double from, const double to,
                             const Eigen::VectorXd& eta) const override {
    auto [mu, sigma] = unpack_(eta);
    return normal_transition_density_(from, to, mu, sigma);
  }

  void KL_(const Eigen::VectorXd& eta, const Eigen::VectorXd& center,
           const Eigen::VectorXd& rho, const double log_s0,
           double& value, Eigen::VectorXd& grad) {
    const double mu = eta(0);
    const double log_s = relative_log_scale_(eta(1), log_s0);
    const double dlog_s = relative_log_scale_derivative_(eta(1));
    const double sigma = scale_from_log_(log_s);
    if (!std::isfinite(mu) || !std::isfinite(log_s) ||
        !std::isfinite(dlog_s) || !std::isfinite(sigma)) {
      set_bad_kl_(eta, value, grad);
      return;
    }
    value = 0.0;
    grad = Eigen::VectorXd::Zero(2);

    double y;
    double logp;
    double w_grad_rho;
    Eigen::Index D = dim();
    Eigen::VectorXd xi(D);
    Eigen::VectorXd grad_logp(D);

    for (Eigen::Index n = 0; n < opts_.N; ++n) {
      const double xn = x_(n);
      const double wn = w_(n);
      y = sigma * xn + mu;
      xi = y * rho + center;
      if (!xi.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      if (!std::isfinite(logp) || !grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      if (!grad_logp.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      value += wn * logp;
      w_grad_rho = wn * grad_logp.dot(rho);
      if (!std::isfinite(w_grad_rho)) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      grad(0) += w_grad_rho;
      grad(1) += w_grad_rho * xn * sigma;
    }
    value += log_s;
    grad(1) += 1.0;
    value = -value;
    grad = -grad;
    grad(1) *= dlog_s;
  }

  static double log_q_(const double x, const double mu,
                       const double sigma) {
    const double z = (x - mu) / sigma;
    return -std::log(sigma) - 0.5 * z * z;
  }

  double normal_transition_density_(const double from, const double to,
                                    const double mu,
                                    const double sigma) const {
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

  std::pair<double, double> unpack_(const Eigen::VectorXd& eta) const {
    const double mu = eta(0);
    const double sigma = scale_from_log_(eta(1));
    return {mu, sigma};
  }

};

} // namespace klhr
