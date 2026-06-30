#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <tuple>

namespace klhr {

class SASKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;

  Eigen::VectorXd fit_sas(const Eigen::VectorXd& rho) {
    return fit_sas_at_(theta_, rho);
  }

  void min_sas_kl(const Eigen::VectorXd& eta,
                  const Eigen::VectorXd& rho,
                  double& value,
                  Eigen::VectorXd& grad) {
    min_sas_kl_at_(eta, theta_, rho, value, grad);
  }

  Eigen::VectorXd sinh_KL_step() {
    return regular_kl_step_();
  }

  Eigen::VectorXd sinh_KL_step(const Eigen::VectorXd& rho) {
    return regular_kl_step_(rho);
  }

protected:
  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    return fit_sas_at_(center, rho);
  }

  double overrelaxed_line_proposal_(const Eigen::VectorXd& eta) override {
    auto [m, s, e] = unpack_sas_(eta);
    return overrelaxed_sas_proposal_(m, s, e);
  }

  double log_line_transition_density_(double from,
                                      double to,
                                      const Eigen::VectorXd& eta) override {
    auto [m, s, e] = unpack_sas_(eta);
    return sas_log_transition_density_(from, to, m, s, e);
  }

  double reference_scale_(const Eigen::VectorXd& eta) override {
    auto [m, s, e] = unpack_sas_(eta);
    (void)m;
    return sas_reference_scale_(s, e);
  }

  Eigen::VectorXd fit_sas_at_(const Eigen::VectorXd& center,
                              const Eigen::VectorXd& rho) {
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
    init << mode.x(0), log_s, 0.0;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      min_sas_kl_at_(eta, center, rho, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 3});
    nfev_ += o.nfev * opts_.N;
    return o.x;
  }

  void min_sas_kl_at_(const Eigen::VectorXd& eta,
                      const Eigen::VectorXd& center,
                      const Eigen::VectorXd& rho,
                      double& value,
                      Eigen::VectorXd& grad) {
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
      xi = t * rho + center;
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      grad_logp = grad_logp.array().min(opts_.grad_clip).max(-opts_.grad_clip);
      line_grad = grad_logp.dot(rho);

      value += wn * (-eta(1) - log_cosh_clipped_(a) - logp);
      grad(0) -= wn * line_grad;
      grad(1) += wn * (-1.0 - line_grad * s * sh);
      grad(2) += wn * (-th - line_grad * s * ch);
    }
  }

  double sas_log_q(const double x, const double m, const double s, const double e) {
    const double z = (x - m) / s;
    const double y = std::asinh(z) - e;
    const double sh = sinh_clipped_(y);

    return -std::log(s) + log_cosh_clipped_(y)
      - 0.5 * sh * sh
      - 0.5 * std::log1p(z * z);
  }

  double sas_log_transition_density_(double from,
                                     double to,
                                     double m,
                                     double s,
                                     double e) {
    s = std::max(s, opts_.tol);
    const double log_density = sas_log_q(to, m, s, e);
    if (opts_.K == 0) {
      return log_density;
    }
    const double u_from = normal_cdf_(sas_to_normal_d1(from, m, s, e));
    const double u_to = normal_cdf_(sas_to_normal_d1(to, m, s, e));
    return ordered_overrelaxed_cdf_log_density_(u_from, u_to) + log_density;
  }

  double overrelaxed_sas_proposal_(double m, double s, double e) {
    s = std::max(s, opts_.tol);
    const double u = normal_cdf_(sas_to_normal_d1(0.0, m, s, e));
    const double up = overrelaxed_cdf_proposal_(u);
    return sas_transform_d1(normal_quantile_(clamp_probability_(up)), m, s, e);
  }

  double sas_transform_d1(double normal_draw, double m, double s, double e) {
    const double z = sinh_clipped_(std::asinh(normal_draw) + e);
    return m + s * z;
  }

  double sas_to_normal_d1(double x, double m, double s, double e) {
    const double z = (x - m) / s;
    return sinh_clipped_(std::asinh(z) - e);
  }

  double sas_reference_scale_(double s, double e) const {
    const double scale = std::max(s, opts_.tol) *
      std::max(cosh_clipped_(e), opts_.tol);
    return std::isfinite(scale) ? scale : opts_.tol;
  }

  std::tuple<double, double, double> unpack_sas_(const Eigen::VectorXd& eta) {
    const double m = eta(0);
    const double c = opts_.scale_clip;
    const double log_s = std::clamp(eta(1), -c, c);
    const double s = std::exp(log_s) + opts_.tol;
    const double e = eta(2);
    return {m, s, e};
  }

  static double log_cosh(const double x) {
    const double ax = std::abs(x);
    return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
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
