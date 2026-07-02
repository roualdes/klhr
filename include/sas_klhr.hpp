#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <tuple>

namespace klhr {

class SASKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;
protected:
    Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                              const Eigen::VectorXd& rho) override {
    Eigen::VectorXd mode_init = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd target_grad = Eigen::VectorXd::Zero(dim());
    auto fg = [&, this](const Eigen::VectorXd& x,
                        double& value, Eigen::VectorXd& g) {
      g.resize(1);
      bsm_.log_density_gradient_noe(x(0) * rho + center, value, target_grad);
      value = -value;
      g(0) = -target_grad.dot(rho);
    };

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init);
    nfev_ += mode.nfev;

    double log_s = 0.0;
    const double h = mode.hess_inv(0, 0);
    if (std::isfinite(h) && h > 0.0) {
      log_s = 0.5 * std::log(h * 1.1);
    }

    Eigen::VectorXd init(3);
    init << mode.x(0), raw_log_scale_(log_s), 0.0;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      KL_(eta, center, rho, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 3});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  double overrelaxed_proposal_(const Eigen::VectorXd& eta) override {
    auto [m, s, e] = unpack_sas_(eta);
    return overrelaxed_sas_proposal_(m, s, e);
  }

  double transition_density_(const double from, const double to,
                             const Eigen::VectorXd& eta) override {
    auto [m, s, e] = unpack_sas_(eta);
    return sas_transition_density_(from, to, m, s, e);
  }

  void KL_(const Eigen::VectorXd& eta, const Eigen::VectorXd& center,
                      const Eigen::VectorXd& rho, double& value, Eigen::VectorXd& grad) {
    auto [m, s, e] = unpack_sas_(eta);
    const double log_s = smooth_log_scale_(eta(1));
    const double dlog_s = smooth_log_scale_derivative_(eta(1));
    value = 0.0;
    grad = Eigen::VectorXd::Zero(3);

    double t;
    double logp;
    double line_grad;
    Eigen::Index D = dim();
    Eigen::VectorXd xi(D);
    Eigen::VectorXd grad_logp(D);

    for (Eigen::Index n = 0; n < opts_.N; ++n) {
      const double xn = x_(n);
      const double wn = w_(n);
      const double a = std::asinh(xn) + e;
      const double sh = sinh_clipped_(a);
      const double ch = cosh_clipped_(a);
      const double th = tanh_clipped_(a);

      t = m + s * sh;
      xi = t * rho + center;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      line_grad = grad_logp.dot(rho);

      value += wn * (-log_s - log_cosh_clipped_(a) - logp);
      grad(0) -= wn * line_grad;
      grad(1) += wn * (-1.0 - line_grad * s * sh);
      grad(2) += wn * (-th - line_grad * s * ch);
    }
    grad(1) *= dlog_s;
  }

  double sas_log_q_(const double x, const double m, const double s, const double e) {
    const double z = (x - m) / s;
    const double y = std::asinh(z) - e;
    const double sh = sinh_clipped_(y);

    return -std::log(s) + log_cosh_clipped_(y)
      - 0.5 * sh * sh
      - 0.5 * std::log1p(z * z);
  }

  double sas_transition_density_(const double from, const double to,
                                 const double m, const double s, const double e) {
    const double ss = std::max(s, opts_.tol);
    const double log_density = sas_log_q_(to, m, ss, e);
    if (opts_.K == 0) {
      return log_density;
    }
    const double u_from = normal_cdf_(Tinv_(from, m, ss, e));
    const double u_to = normal_cdf_(Tinv_(to, m, ss, e));
    return overrelaxed_density_(u_from, u_to) + log_density;
  }

  double overrelaxed_sas_proposal_(const double m, const double s, const double e) {
    const double ss = std::max(s, opts_.tol);
    const double u = normal_cdf_(Tinv_(0.0, m, ss, e));
    const double up = overrelaxed_proposal_impl_(u);
    return T_(normal_quantile_(clamp_probability_(up)), m, ss, e);
  }

  double T_(const double normal_draw, const double m, const double s, const double e) {
    const double z = sinh_clipped_(std::asinh(normal_draw) + e);
    return m + s * z;
  }

  double Tinv_(const double x, const double m, const double s, const double e) {
    const double z = (x - m) / s;
    return sinh_clipped_(std::asinh(z) - e);
  }

  std::tuple<double, double, double> unpack_sas_(const Eigen::VectorXd& eta) {
    const double m = eta(0);
    const double log_s = smooth_log_scale_(eta(1));
    const double s = std::exp(log_s) + opts_.tol;
    const double e = eta(2);
    return {m, s, e};
  }

  static double log_cosh_(const double x) {
    const double ax = std::abs(x);
    return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
  }

  double scale_clipped_(const double x) const {
    const double c = opts_.scale_clip;
    return std::clamp(x, -c, c);
  }

  double sinh_clipped_(const double x) const {
    return std::sinh(scale_clipped_(x));
  }

  double cosh_clipped_(const double x) const {
    return std::cosh(scale_clipped_(x));
  }

  double tanh_clipped_(const double x) const {
    return std::tanh(scale_clipped_(x));
  }

  double log_cosh_clipped_(const double x) const {
    return log_cosh_(scale_clipped_(x));
  }

  double smooth_log_scale_(const double raw) const {
    const double c = opts_.scale_clip;
    if (!std::isfinite(c) || c <= 0.0) {
      return raw;
    }
    if (!std::isfinite(raw)) {
      return std::isnan(raw) ? 0.0 : std::copysign(c, raw);
    }
    return c * std::tanh(raw / c);
  }

  double smooth_log_scale_derivative_(const double raw) const {
    const double c = opts_.scale_clip;
    if (!std::isfinite(c) || c <= 0.0) {
      return 1.0;
    }
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    const double th = std::tanh(raw / c);
    return 1.0 - th * th;
  }

  double raw_log_scale_(const double log_s) const {
    const double c = opts_.scale_clip;
    if (!std::isfinite(c) || c <= 0.0) {
      return log_s;
    }
    if (!std::isfinite(log_s)) {
      return 0.0;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const double z = std::clamp(log_s / c, -1.0 + eps, 1.0 - eps);
    return c * std::atanh(z);
  }
};

} // namespace klhr
