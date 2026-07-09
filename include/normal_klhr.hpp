#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace klhr {

class NormalKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;

  const std::vector<double>& normal_mean_history() const {
    return normal_mean_;
  }

  const std::vector<double>& normal_standard_deviation_history() const {
    return normal_standard_deviation_;
  }

protected:
  void record_kl_step_(const Eigen::VectorXd& eta, const double xi,
                       const bool accepted) override {
    (void) xi;
    (void) accepted;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const bool valid_eta = eta.size() >= 2 && eta.allFinite();
    const double sigma = valid_eta ? scale_from_log_(eta(1)) : nan;

    normal_mean_.push_back(valid_eta ? eta(0) : nan);
    normal_standard_deviation_.push_back(
      std::isfinite(sigma) ? sigma : nan);
  }

  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    const LineModeEstimate mode = fit_line_mode_(center, rho);
    Eigen::VectorXd init(2);
    init << mode.mode, 0.0;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      KL_(eta, center, rho, mode.log_scale, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 4});
    nfev_ += o.nfev * opts_.N;
    const Eigen::VectorXd raw =
      o.x.size() == 2 && o.x.allFinite() ? o.x : init;
    Eigen::VectorXd out(2);
    out << raw(0), relative_log_scale_(raw(1), mode.log_scale);
    return out;
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

    // TODO openmp it up, try changing N
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
    const double sigma = scale_from_log_(eta(1));
    return {mu, sigma};
  }

  std::vector<double> normal_mean_;
  std::vector<double> normal_standard_deviation_;
};

} // namespace klhr
