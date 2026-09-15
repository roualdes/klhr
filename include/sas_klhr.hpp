#pragma once

#include "base_klhr.hpp"
#include "normal_quantile.hpp"

#include <tuple>

namespace klhr {

class SASKLHR : public BaseKLHR {
public:
  using BaseKLHR::BaseKLHR;

protected:
  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    return fit_line_with_kl_fallback_(
      center, rho, 3,
      [this, &center, &rho](const Eigen::VectorXd& eta,
                            const double log_scale,
                            double& value, Eigen::VectorXd& grad) {
        KL_(eta, center, rho, log_scale, value, grad);
      },
      [this](const Eigen::VectorXd& raw, const double log_scale) {
        Eigen::VectorXd eta(3);
        eta << raw(0), relative_log_scale_(raw(1), log_scale),
          bounded_skew_(raw(2));
        return eta;
      });
  }

  double overrelaxed_proposal_(const Eigen::VectorXd& eta,
                               const double from) override {
    const auto [m_, s, e_] = unpack_sas_(eta);
    const double m = m_;
    const double e = e_;
    const double ss = std::max(s, opts_.tol);
    return overrelaxed_proposal_from_cdf_(
      normal_cdf_(Tinv_(from, m, ss, e)),
      [this, m, ss, e](const double z) { return T_(z, m, ss, e); });
  }

  double log_line_density_(const double t,
                           const Eigen::VectorXd& eta) const override {
    auto [m, s, e] = unpack_sas_(eta);
    return sas_log_q_(t, m, std::max(s, opts_.tol), e);
  }

  void KL_(const Eigen::VectorXd& eta,
           const Eigen::VectorXd& center,
           const Eigen::VectorXd& rho,
           const double log_s0,
           double& value,
           Eigen::VectorXd& grad) {
    const double m = eta(0);
    const double log_s = relative_log_scale_(eta(1), log_s0);
    const double dlog_s = relative_log_scale_derivative_(eta(1));
    const double s = scale_from_log_(log_s);
    const double e = bounded_skew_(eta(2));
    const double de = bounded_skew_derivative_(eta(2));
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
      if (!xi.allFinite()) {
        set_bad_kl_(eta, value, grad);
        return;
      }
      bsm_.log_density_gradient_noe(xi, logp, grad_logp);
      line_grad = grad_logp.dot(rho);
      // Covers everything KL_ consumes: only the projection of grad_logp on
      // rho enters the objective, and a NaN or infinity anywhere in the vector
      // reaches line_grad. See the matching note in NormalKLHR::KL_.
      if (!std::isfinite(logp) || !std::isfinite(line_grad)) {
        set_bad_kl_(eta, value, grad);
        return;
      }

      value += wn * (-log_s - log_cosh_clipped_(a) - logp);
      grad(0) -= wn * line_grad;
      grad(1) += wn * (-1.0 - line_grad * s * sh);
      grad(2) += wn * (-th - line_grad * s * ch);
    }
    grad(1) *= dlog_s;
    grad(2) *= de;
  }

  double sas_log_q_(const double x, const double m, const double s,
                    const double e) const {
    const double z = (x - m) / s;
    const double y = std::asinh(z) - e;
    const double sh = sinh_clipped_(y);

    return -std::log(s) + log_cosh_clipped_(y)
      - 0.5 * sh * sh
      - 0.5 * std::log1p(z * z);
  }

  double T_(const double normal_draw, const double m, const double s,
            const double e) const {
    const double z = sinh_clipped_(std::asinh(normal_draw) + e);
    return m + s * z;
  }

  double Tinv_(const double x, const double m, const double s,
               const double e) const {
    const double z = (x - m) / s;
    return sinh_clipped_(std::asinh(z) - e);
  }

  std::tuple<double, double, double>
  unpack_sas_(const Eigen::VectorXd& eta) const {
    const double m = eta(0);
    const double s = scale_from_log_(eta(1));
    const double e = eta(2);
    return {m, s, e};
  }

  static double log_cosh_(const double x) {
    const double ax = std::abs(x);
    return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
  }

  double sas_arg_clipped_(const double x) const {
    const double c = opts_.sas_arg_clip;
    return std::clamp(x, -c, c);
  }

  double sinh_clipped_(const double x) const {
    return std::sinh(sas_arg_clipped_(x));
  }

  double cosh_clipped_(const double x) const {
    return std::cosh(sas_arg_clipped_(x));
  }

  double tanh_clipped_(const double x) const {
    return std::tanh(sas_arg_clipped_(x));
  }

  double log_cosh_clipped_(const double x) const {
    return log_cosh_(sas_arg_clipped_(x));
  }

  static double bounded_skew_(const double raw) {
    return numerics::bounded_reparam(raw, skew_radius_());
  }

  static double bounded_skew_derivative_(const double raw) {
    return numerics::bounded_reparam_derivative(raw, skew_radius_());
  }

  static constexpr double skew_radius_() {
    return 5.0;
  }

};

} // namespace klhr
