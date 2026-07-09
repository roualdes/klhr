#pragma once

#include "base_klhr.hpp"
#include "bfgs.hpp"
#include "normal_quantile.hpp"

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace klhr {

class SASKLHR : public BaseKLHR {
public:
  SASKLHR(std::string stan_file, std::string json_file,
          const KlhrOptions& options = KlhrOptions{}) :
    BaseKLHR(stan_file, json_file, options) {}

  const std::vector<double>& sas_location_history() const {
    return sas_location_;
  }

  const std::vector<double>& sas_scale_history() const {
    return sas_scale_;
  }

  const std::vector<double>& sas_skew_history() const {
    return sas_skew_;
  }

  const std::vector<double>& sas_m_history() const {
    return sas_location_;
  }

  const std::vector<double>& sas_xi_history() const {
    return sas_xi_;
  }

  const std::vector<double>& sas_accepted_xi_history() const {
    return sas_accepted_xi_;
  }

  const std::vector<double>& sas_accepted_history() const {
    return sas_accepted_;
  }

protected:
  void record_kl_step_(const Eigen::VectorXd& eta, const double xi,
                       const bool accepted) override {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const bool valid_eta = eta.size() >= 3 && eta.allFinite();
    const bool valid_xi = std::isfinite(xi);
    const double scale = valid_eta ? scale_from_log_(eta(1)) : nan;

    sas_location_.push_back(valid_eta ? eta(0) : nan);
    sas_scale_.push_back(std::isfinite(scale) ? scale : nan);
    sas_skew_.push_back(valid_eta ? eta(2) : nan);
    sas_xi_.push_back(valid_xi ? xi : nan);
    sas_accepted_xi_.push_back(accepted && valid_xi ? xi : nan);
    sas_accepted_.push_back(accepted ? 1.0 : 0.0);
  }

  Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                            const Eigen::VectorXd& rho) override {
    const LineModeEstimate mode = fit_line_mode_(center, rho);
    Eigen::VectorXd init(3);
    init << mode.mode, 0.0, 0.0;

    auto kl = [&, this](const Eigen::VectorXd& eta,
                        double& value, Eigen::VectorXd& grad) {
      KL_(eta, center, rho, mode.log_scale, value, grad);
    };
    bfgs::BfgsResult o =
      bfgs::bfgs(kl, init, {.gtol = opts_.gtol,
                            .xrtol = opts_.gtol,
                            .maxiter_bfgs = 3});
    nfev_ += o.nfev * opts_.N;
    const Eigen::VectorXd raw =
      o.x.size() == 3 && o.x.allFinite() ? o.x : init;
    Eigen::VectorXd out(3);
    out << raw(0), relative_log_scale_(raw(1), mode.log_scale),
      bounded_skew_(raw(2));
    return out;
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
                      const Eigen::VectorXd& rho, const double log_s0,
                      double& value, Eigen::VectorXd& grad) {
    const double m = eta(0);
    const double log_s = relative_log_scale_(eta(1), log_s0);
    const double dlog_s = relative_log_scale_derivative_(eta(1));
    const double s = scale_from_log_(log_s);
    const double e = bounded_skew_(eta(2));
    const double de = bounded_skew_derivative_(eta(2));
    if (!std::isfinite(m) || !std::isfinite(log_s) ||
        !std::isfinite(dlog_s) || !std::isfinite(s) ||
        !std::isfinite(e) || !std::isfinite(de)) {
      set_bad_kl_(eta, value, grad);
      return;
    }
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
      if (!std::isfinite(t) || !xi.allFinite()) {
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
      line_grad = grad_logp.dot(rho);
      if (!std::isfinite(line_grad)) {
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

  double bounded_skew_(const double raw) const {
    const double r = skew_radius_();
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    return r * std::tanh(raw / r);
  }

  double bounded_skew_derivative_(const double raw) const {
    const double r = skew_radius_();
    if (!std::isfinite(raw)) {
      return 0.0;
    }
    const double th = std::tanh(raw / r);
    return 1.0 - th * th;
  }

  static constexpr double skew_radius_() {
    return 5.0;
  }

  std::vector<double> sas_location_;
  std::vector<double> sas_scale_;
  std::vector<double> sas_skew_;
  std::vector<double> sas_xi_;
  std::vector<double> sas_accepted_xi_;
  std::vector<double> sas_accepted_;
};

} // namespace klhr
