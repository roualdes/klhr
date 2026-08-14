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

  double overrelaxed_proposal_(const Eigen::VectorXd& eta,
                               const double from) override {
    const auto [mu_, sigma] = unpack_(eta);
    const double mu = mu_;
    const double s = std::max(sigma, opts_.tol);
    return overrelaxed_proposal_from_cdf_(
      normal_cdf_((from - mu) / s),
      [mu, s](const double z) { return mu + s * z; });
  }

  double log_line_density_(const double t,
                           const Eigen::VectorXd& eta) const override {
    auto [mu, sigma] = unpack_(eta);
    return log_q_(t, mu, std::max(sigma, opts_.tol));
  }

  void KL_(const Eigen::VectorXd& eta, const Eigen::VectorXd& center,
           const Eigen::VectorXd& rho, const double log_s0,
           double& value, Eigen::VectorXd& grad) {
    const double mu = eta(0);
    const double log_s = relative_log_scale_(eta(1), log_s0);
    const double dlog_s = relative_log_scale_derivative_(eta(1));
    const double sigma = scale_from_log_(log_s);
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
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      w_grad_rho = wn * grad_logp.dot(rho);
      // This covers everything KL_ consumes. The only part of grad_logp that
      // reaches the objective is its projection on rho, and a NaN or an
      // infinity anywhere in the vector reaches w_grad_rho: Eigen's min/max
      // propagate NaN in this argument order, and an infinite component either
      // survives the dot product or meets a zero rho_d and turns it NaN. There
      // is deliberately no componentwise check, because a garbage component
      // orthogonal to rho cannot affect the fit.
      //
      // One caveat, and it belongs to the clip rather than to the check: with
      // a finite grad_clip an infinite gradient is clipped to +/-grad_clip
      // *before* this line, so the point then looks feasible. The default
      // grad_clip is infinite, which leaves infinities intact.
      if (!std::isfinite(logp) || !std::isfinite(w_grad_rho)) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      value += wn * logp;
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

  std::pair<double, double> unpack_(const Eigen::VectorXd& eta) const {
    const double mu = eta(0);
    const double sigma = scale_from_log_(eta(1));
    return {mu, sigma};
  }

};

} // namespace klhr
