#pragma once

#include "bfgs.hpp"
#include "diagonal_metric.hpp"
#include "gausshermite.hpp"
#include "klhr_numerics.hpp"
#include "normal_quantile.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

namespace klhr {

struct KlhrOptions {
  std::uint64_t seed = 0;
  Eigen::Index N = 8;
  double tol = 1e-10;
  double sas_arg_clip = 30;
  double gtol = 1e-5;
  // Accept the Laplace line fit when its KL gradient residual (location
  // measured in proposal standard deviations) is below this. Tightening it
  // makes the KL optimizer run far more often: on funnel that roughly
  // doubled cost with no accuracy gain, and on an exactly Gaussian target
  // the residual is ~0 so the setting has no effect at all.
  double laplace_kl_residual_tol = 1.0;
  std::size_t maxiter_bfgs = 32;
  // Ordered overrelaxation level. Fixed for the whole run; odd only.
  std::size_t K = 15;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
};

class BaseKLHR {
public:
  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;

  BaseKLHR(std::string stan_file, std::string json_file,
           const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(normalized_options_(options)),
    metric_(bsm_.dim(), opts_.warmup, opts_.windowsize, opts_.windowscale,
            opts_.tol) {

    if (opts_.seed == 0) {
      std::random_device rd;
      std::uint64_t r1 = rd();
      std::uint64_t r2 = rd();
      opts_.seed = (r1 << 32) ^ r2;
      if (opts_.seed == 0) {
        opts_.seed = 1;
      }
      rng_.seed(opts_.seed);
    }

    std::uniform_int_distribution<unsigned int> uniform_uint;
    mcmcpp::bsrng bsrng = bsm_.make_rng(uniform_uint(rng_));
    theta_ = bsm_.param_initialize(bsrng);
    if (!theta_.allFinite()) {
      throw std::runtime_error("BaseKLHR: invalid initial state");
    }

    w_.resize(opts_.N);
    x_.resize(opts_.N);
    gauss_hermite(opts_.N, w_, x_);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);

    metric_.seed_mean(theta_);

    nfev_ = 0;
    acceptance_rate_ = 0.0;
    log_density_ = bsm_.log_density_noe(theta_);
    ++nfev_;
    // Defensive: param_initialize already retries until the density is
    // finite. A non-finite start would reject on every draw, silently.
    if (!std::isfinite(log_density_)) {
      throw std::runtime_error(
        "BaseKLHR: initial log density is not finite");
    }
    draw_ = 0;
  }

  virtual ~BaseKLHR() = default;

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  Eigen::VectorXd metric_variance() const { return metric_.variance(); }

  std::size_t overrelaxation_K() const {
    return opts_.K;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    const Eigen::VectorXd rho = random_direction();
    kl_step_(rho);
    metric_.adapt(theta_, draw_);
    return bsm_.param_constrain(theta_);
  }

  Eigen::VectorXd random_direction() {
    return metric_.random_direction(rng_, std_normal_);
  }

protected:
  // Fit the line-conditional approximation, in coordinates relative to
  // `center` (a canonical point on the line).
  virtual Eigen::VectorXd fit_line_(const Eigen::VectorXd& center,
                                    const Eigen::VectorXd& rho) = 0;

  // Ordered overrelaxation of `from` with respect to the fitted density.
  virtual double overrelaxed_proposal_(const Eigen::VectorXd& eta,
                                       const double from) = 0;

  virtual double log_line_density_(const double t,
                                   const Eigen::VectorXd& eta) const = 0;

  struct LineModeEstimate {
    double mode = 0.0;
    double log_scale = 0.0;
    bool curvature_usable = false;
    bool success = false;
  };

  LineModeEstimate fit_line_mode_(const Eigen::VectorXd& center,
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

    bfgs::BfgsResult mode = bfgs::bfgs(fg, mode_init, bfgs_options_());
    nfev_ += mode.nfev;

    LineModeEstimate out;
    if (mode.x.size() == 1 && std::isfinite(mode.x(0))) {
      out.mode = mode.x(0);
    }
    double inverse_curvature = std::numeric_limits<double>::quiet_NaN();
    if (mode.hess_inv.rows() == 1 && mode.hess_inv.cols() == 1) {
      inverse_curvature = mode.hess_inv(0, 0);
    }
    out.curvature_usable =
      std::isfinite(inverse_curvature) && inverse_curvature > 0.0;
    if (out.curvature_usable) {
      out.log_scale = 0.5 * std::log(inverse_curvature);
    }
    out.success = mode.success;
    return out;
  }

  bfgs::BfgsOptions bfgs_options_() const {
    return {.gtol = opts_.gtol,
            .xrtol = opts_.gtol,
            .maxiter_bfgs = opts_.maxiter_bfgs};
  }

  template <typename EvaluateKl, typename TransformParameters>
  Eigen::VectorXd fit_line_with_kl_fallback_(
      const Eigen::VectorXd& center,
      const Eigen::VectorXd& rho,
      const Eigen::Index parameter_count,
      EvaluateKl evaluate_kl,
      TransformParameters transform_parameters) {
    const LineModeEstimate mode = fit_line_mode_(center, rho);

    Eigen::VectorXd init = Eigen::VectorXd::Zero(parameter_count);
    init(0) = mode.mode;

    auto kl = [&](const Eigen::VectorXd& eta,
                  double& value, Eigen::VectorXd& grad) {
      evaluate_kl(eta, mode.log_scale, value, grad);
    };

    if (!mode.success || !mode.curvature_usable) {
      bfgs::BfgsResult fit = bfgs::bfgs(kl, init, bfgs_options_());
      nfev_ += fit.nfev * opts_.N;
      const Eigen::VectorXd raw =
        fit.x.size() == parameter_count && fit.x.allFinite() ? fit.x : init;
      return transform_parameters(raw, mode.log_scale);
    }

    double initial_value = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd initial_grad;
    kl(init, initial_value, initial_grad);
    nfev_ += opts_.N;

    const bool valid_initial =
      std::isfinite(initial_value) &&
      !numerics::is_bad_kl(initial_value) &&
      initial_grad.size() == parameter_count &&
      initial_grad.allFinite();
    if (valid_initial) {
      Eigen::VectorXd residual = initial_grad;
      // Express the location residual in proposal-standard-deviation units.
      residual(0) *= scale_from_log_(mode.log_scale);
      if (residual.lpNorm<Eigen::Infinity>() <=
          opts_.laplace_kl_residual_tol) {
        return transform_parameters(init, mode.log_scale);
      }
    }

    // Reuse the validation evaluation as BFGS's initial evaluation.
    bool initial_evaluation_available = true;
    auto cached_kl = [&](const Eigen::VectorXd& eta,
                         double& value, Eigen::VectorXd& grad) {
      const bool at_initial =
        eta.size() == init.size() &&
        (eta.array() == init.array()).all();
      if (initial_evaluation_available && at_initial) {
        initial_evaluation_available = false;
        value = initial_value;
        grad = initial_grad;
        return;
      }
      kl(eta, value, grad);
    };

    bfgs::BfgsResult fit = bfgs::bfgs(cached_kl, init, bfgs_options_());
    // The cached first evaluation was already charged above.
    nfev_ += (fit.nfev > 0 ? fit.nfev - 1 : 0) * opts_.N;
    const Eigen::VectorXd raw =
      fit.x.size() == parameter_count && fit.x.allFinite() ? fit.x : init;
    return transform_parameters(raw, mode.log_scale);
  }

  // K=1 is the independence kernel; identical to K=0. Even K adds an
  // unmodelled self-transition atom at the middle rank, so round down to the
  // nearest usable odd level.
  static std::size_t force_odd_K_(const std::size_t K) {
    if (K <= 1) {
      return 0;
    }
    return K % 2 == 0 ? K - 1 : K;
  }

  static KlhrOptions normalized_options_(KlhrOptions options) {
    options.N = std::max<Eigen::Index>(1, options.N);
    if (!(options.tol > 0.0) || !std::isfinite(options.tol)) {
      options.tol = 1e-10;
    }
    if (!(options.sas_arg_clip > 0.0) ||
        !std::isfinite(options.sas_arg_clip)) {
      options.sas_arg_clip = 30.0;
    }
    // Fallbacks match the declared defaults; a rejected value must land on the
    // documented setting, not on a different one.
    if (!(options.gtol > 0.0) || !std::isfinite(options.gtol)) {
      options.gtol = KlhrOptions{}.gtol;
    }
    if (!(options.laplace_kl_residual_tol >= 0.0) ||
        !std::isfinite(options.laplace_kl_residual_tol)) {
      options.laplace_kl_residual_tol = KlhrOptions{}.laplace_kl_residual_tol;
    }
    options.maxiter_bfgs = std::max<std::size_t>(1, options.maxiter_bfgs);
    const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
    options.K = force_odd_K_(std::min(options.K, int_max));
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    return options;
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  DiagonalMetric metric_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights

  std::size_t draw_;
  std::size_t kl_steps_ = 0;

  // Ordered overrelaxation is reversible with respect to the distribution it
  // is applied to (Neal 1995, section 4.2). Fitting one density per *line*
  // rather than one per endpoint therefore cancels the overrelaxation kernel
  // out of the Hastings ratio entirely, leaving the importance ratio
  // pi(t1)q(t0) / pi(t0)q(t1). That needs the fit to be a function of the
  // line alone, so the fit is centred on the projection of the adaptation
  // mean, which is the same point whichever end of the line we stand at.
  void kl_step_(const Eigen::VectorXd& rho) {
    auto update_acceptance = [this](const bool accepted) {
      ++kl_steps_;
      const double d = accepted - acceptance_rate_;
      acceptance_rate_ += d / static_cast<double>(kl_steps_);
    };

    const double t0 = line_coordinate_(theta_, rho);
    if (!std::isfinite(t0)) {
      update_acceptance(false);
      return;
    }
    const Eigen::VectorXd center = theta_ - t0 * rho;

    const Eigen::VectorXd eta = fit_line_(center, rho);
    const double t1 = overrelaxed_proposal_(eta, t0);
    const double xi = t1 - t0;
    if (!std::isfinite(xi)) {
      update_acceptance(false);
      return;
    }
    const Eigen::VectorXd thetap = xi * rho + theta_;
    if (!thetap.allFinite()) {
      update_acceptance(false);
      return;
    }

    const double ldp = bsm_.log_density_noe(thetap);
    ++nfev_;
    if (!std::isfinite(ldp)) {
      update_acceptance(false);
      return;
    }

    const double log_q0 = log_line_density_(t0, eta);
    const double log_q1 = log_line_density_(t1, eta);
    if (!std::isfinite(log_q0) || !std::isfinite(log_q1)) {
      update_acceptance(false);
      return;
    }

    const double a = ldp - log_density_ + log_q0 - log_q1;
    if (!std::isfinite(a)) {
      update_acceptance(false);
      return;
    }
    const double log_u = std::log(std_uniform_(rng_));
    const bool accepted = a >= 0.0 || log_u < a;
    update_acceptance(accepted);
    if (accepted) {
      theta_ = thetap;
      log_density_ = ldp;
    }
  }

  // Position of theta along the line, measured from the projection of the
  // adaptation mean. Depends only on the line and on frozen adaptation
  // state, never on which endpoint is current.
  double line_coordinate_(const Eigen::VectorXd& theta,
                          const Eigen::VectorXd& rho) const {
    return (theta - metric_.mean()).dot(rho);
  }

  void set_bad_kl_(const Eigen::VectorXd& eta, double& value,
                   Eigen::VectorXd& grad) const {
    numerics::set_bad_kl(eta, value, grad);
  }

  double relative_log_scale_(const double raw, const double log_s0) const {
    return numerics::relative_log_scale(raw, log_s0);
  }

  double relative_log_scale_derivative_(const double raw) const {
    return numerics::relative_log_scale_derivative(raw);
  }

  double scale_from_log_(double log_s) const {
    return numerics::scale_from_log(log_s, opts_.tol);
  }

  // Ordered overrelaxation given a position u_raw under the fitted CDF, with
  // the trip back from a standard normal draw to the line coordinate supplied
  // by the caller.
  //
  // Every branch here has to be reversible with respect to q, because kl_step_
  // cancels the proposal kernel out of the Hastings ratio on exactly that
  // assumption.
  template <typename ToCoordinate>
  double overrelaxed_proposal_from_cdf_(const double u_raw,
                                        ToCoordinate to_coordinate) {
    if (numerics::probability_saturated(u_raw)) {
      // `from` lies past the last quantile of q that a double can distinguish,
      // so u_raw says nothing usable about where it is and the overrelaxation
      // kernel would be applied to a clamped surrogate. Fall back to an
      // independence draw from q, which is exactly reversible for any K.
      //
      // Selecting the fallback on a state-dependent condition does break
      // detailed balance in principle -- from a saturated t0 the kernel is
      // q(.), while the reverse move from an unsaturated t1 would overrelax.
      // Proposing `from` itself was tried on that reasoning and is worse:
      // the imbalance is not repaired either way (the flux mismatch stays
      // O(q(saturated)), about 1e-16, since the set is |z| > 8.15 under q),
      // but the identity also makes the set absorbing, so a chain that
      // reaches it can never leave. The independence draw is the escape
      // hatch that makes this branch a rescue rather than a trap.
      return to_coordinate(
        normal_quantile(clamp_probability_(std_uniform_(rng_))));
    }
    const double up = overrelaxed_proposal_impl_(clamp_probability_(u_raw));
    return to_coordinate(normal_quantile(clamp_probability_(up)));
  }

  double overrelaxed_proposal_impl_(const double u) {
    // K = 0 is the independence kernel. The choice is a fixed property of the
    // configuration rather than of the current state, so it is q-reversible.
    if (opts_.K == 0) {
      return clamp_probability_(std_uniform_(rng_));
    }

    const int K = static_cast<int>(opts_.K);
    std::binomial_distribution<int> binomial(K, u);
    const int r = binomial(rng_);

    double up = u;
    if (r > K - r) {
      const double v = beta_rng_(K - r + 1.0, 2.0 * r - K);
      up = u * v;
    } else if (r < K - r) {
      const double v = beta_rng_(r + 1.0, K - 2.0 * r);
      up = 1.0 - (1.0 - u) * v;
    }

    return clamp_probability_(up);
  }

  double beta_rng_(double a, double b) {
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);

    double x = gamma_a(rng_);
    double y = gamma_b(rng_);
    double total = x + y;
    while (!std::isfinite(total) || total <= 0.0) {
      x = gamma_a(rng_);
      y = gamma_b(rng_);
      total = x + y;
    }
    return x / total;
  }

  static double clamp_probability_(double p) {
    return numerics::clamp_probability(p);
  }

  static double normal_cdf_(double z) {
    return normal_cdf(z);
  }

};

} // namespace klhr
