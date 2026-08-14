#pragma once

#include "bfgs.hpp"
#include "gausshermite.hpp"
#include "k_adaptation.hpp"
#include "klhr_numerics.hpp"
#include "normal_quantile.hpp"
#include "onlinepca.hpp"
#include "reflected_transport.hpp"
#include "sketch.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>
#include <welford.hpp>
#include <windowedadaptation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace klhr {

inline constexpr double positive_infinity =
  std::numeric_limits<double>::infinity();

struct KlhrOptions {
  std::uint64_t seed = 0;
  Eigen::Index N = 8;
  double tol = 1e-10;
  double grad_clip = positive_infinity;
  double sas_arg_clip = 30;
  double gtol = 1e-5;
  // Accept the Laplace line fit when its KL gradient residual (location
  // measured in proposal standard deviations) is below this. Tightening it
  // makes the KL optimizer run far more often: on funnel that roughly
  // doubled cost with no accuracy gain, and on an exactly Gaussian target
  // like ar1 the residual is ~0 so the setting has no effect at all.
  double laplace_kl_residual_tol = 1.0;
  std::size_t maxiter_bfgs = 32;
  std::size_t transport_maxiter_bfgs = 8;
  std::size_t K = 15;  // Odd K only
  bool adapt_K = true;
  std::size_t K_max = 511;
  std::size_t K_windowsize = 50;
  std::size_t K_windowscale = 2;
  std::size_t K_refresh_lags = 4;
  bool K_interleave_trials = true;
  std::size_t warmup = 1'000;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  Eigen::Index J = 10;
  double direction_lowrank_weight = 1.0;
  double direction_min_diag_fraction = 0.1;
  bool lowrank_during_warmup = true;
  double pca_freeze_fraction = 0.1;
  // Window closures required before the OnlinePCA low-rank direction is used.
  // Inert under the default mixture_direction = true, which routes adaptation
  // to the sketch instead: projected_moments_ never accumulates, so
  // activate_projected_pair_ returns early every time and the count never
  // leaves zero. It is still a live --lowrank-min-activations flag in
  // examples/example.cpp, so hardcoding it means removing that flag too -- and
  // the honest version of that change removes the whole OnlinePCA path, which
  // the default configuration no longer reaches.
  std::size_t lowrank_min_activations = 1;
  Eigen::Index sketch_columns = 60;
  Eigen::Index transport_J = 1;
  // Draw the direction from a mixture of a diagonal, a sketched covariance
  // and a trajectory-curvature component, with the weights adapted by a
  // bandit. Replaces the single global choice of direction law: which law is
  // best is target dependent, so the mixture settles it per run instead of
  // per build.
  //
  // On by default: DIAG + COV + HESS with the native gate beats the
  // diagonal-only sampler on five of six models, decisively where there is
  // correlation to find (ar1 0.173x, corr-normal 0.114x RMSE-mean, the
  // latter with 15.7x the mean squared jump of DIAG + COV alone). It pays
  // ~3.5% on ill-normal and ~5% on normal, where a covariance component has
  // nothing to find and the bandit can only shrink it toward its floor.
  bool mixture_direction = true;
  // Cap on the sketch component's rank; 0 leaves the spectrum untruncated.
  //
  // The cap is load-bearing, which is the opposite of what the "a rank-J
  // truncation is a lossy compression of something you already have exactly"
  // argument predicts. That argument assumes the whole spectrum is estimable.
  // At ESS/D around 0.26 it is not: the leading few directions are, the tail
  // is noise, and sampling along the tail means proposing in directions the
  // chain has not resolved. On corr-normal, uncapping cost ~25% in
  // standardized RMSE (0.132 -> 0.170 unwhitened, 0.140 -> 0.166 whitened).
  Eigen::Index sketch_max_rank = 20;
  // Gate the Hessian component on its own Rayleigh-Ritz spectrum.
  //
  // Rayleigh-Ritz, concretely: the refresh already holds an orthonormal Q
  // spanning the subspace its probes reached, and `small` = Q' Sigma_w Q is
  // the whitened covariance restricted to that subspace. Its eigenvalues are
  // the Ritz values -- the best estimates of Sigma_w's own eigenvalues
  // obtainable from that subspace -- and the power iterations are what tilt Q
  // toward the leading ones, so the top Ritz value tracks the true top
  // eigenvalue closely while the rest of the spectrum is only sampled. The
  // reading taken is the top Ritz value over the median one.
  //
  // That ratio gates because it is the question the component answers. A HESS
  // draw proposes along the leading curvature directions at their own
  // magnitudes, which beats the diagonal only if those directions really are
  // much wider than typical. Whitening has already divided the diagonal out,
  // so a ratio near one says the whitened covariance is near isotropic and
  // there is no anisotropy left for a low-rank term to exploit -- the metric
  // would be picking up noise.
  //
  // Two questions have to be answered about this component: can it be built,
  // and is it worth building. CG failure already answers the first for free --
  // a failed refresh leaves it out of live_components_ - but says nothing
  // about the second: on ill-normal and normal CG succeeds 3/3 and the
  // component is still harmful. The anisotropy of Q'Sigma_w Q answers the
  // second, is computed inside the refresh at no cost, and separates the
  // models where CG succeeds: ar1 9.8 and corr-normal 28.4 against ill-normal
  // 2.7 and normal 2.2.
  //
  // It misreads funnel (55.0) and earnings (1.3), but those are exactly the
  // models where CG fails anyway (0.7/8 and 1.0/8), so the capability gate
  // already covers them.
  //
  // There is also no bootstrapping problem: the reading exists exactly when
  // the component does, since both require a successful refresh.
  //
  // This replaced a gate on the whitened anisotropy of an (s, y) secant block,
  // which was measured indistinguishable from it -- 0 of 18 cells resolved
  // across six models and three metrics -- and cost a gradient evaluation per
  // stride, 6.8% of the total on earnings. The secant machinery was deleted
  // once nothing read it. "native" is retained in the name to keep it
  // unambiguous against that older, differently-scaled threshold.
  double hess_gate_native_threshold = 5.0;
  // What to do before the first usable anisotropy reading arrives: true admits
  // the gated component and withdraws it once the statistic disagrees, false
  // withholds it until the statistic vouches for it. A reading exists from the
  // first successful Hessian refresh onwards, so this governs only the windows
  // before that -- and every window of a run where no refresh ever succeeds.
  // Failing
  // closed cost 17% on ar1 and 14% on corr-normal -- both resolved -- because
  // the component sat out the early windows on targets where the gate ends
  // open on every seed. The exposure the other way is bounded by what a
  // useless component costs for those same few windows, and full exposure
  // over a whole run is only ~7% (ill-normal 1.066, normal 1.087). So the
  // asymmetry favours admitting the component and withdrawing it once there
  // is evidence, rather than withholding it until there is.
  bool anisotropy_gate_fail_open = true;
  // Mixture floors. The diagonal floor is structural, and deliberately not
  // small: on a target with no correlation to find (ill-normal) the diagonal
  // is the correct answer, standardized ESJD is exactly flat so the bandit
  // gets no gradient, and the floor is what keeps a useless component from
  // holding weight indefinitely.
  double mixture_floor_diagonal = 0.25;
  double mixture_floor_sketch = 0.02;
  double mixture_floor_hess = 0.02;
  // Log-odds prior the weights relax toward when rewards are uninformative.
  double mixture_prior_sketch = -1.0;
  double mixture_prior_hess = -1.0;
  double bandit_gamma0 = 0.5;
  double bandit_decay = 0.6;
  // Pull toward the prior. At 0.1 the cumulative pull over a run was about
  // 0.2 in log-odds against a prior of -1, so an uninformative component kept
  // most of its starting mass: on ill-normal the diagonal fired only 45% of
  // the time and the mixture came out 42% worse than diagonal-only. The
  // reward term is O(1) per update, so a genuinely useful component still
  // overcomes this easily.
  double bandit_shrink = 0.5;
  std::size_t bandit_min_draws = 20;
  // Terminal buffer: the final warmup draws, over which every slow and fast
  // parameter is frozen so the chain equilibrates to the kernel that will
  // actually sample. Without it the last metric change lands on the final
  // warmup draw.
  std::size_t terminal_buffer = 50;
  Eigen::Index hessian_oversample = 10;
  std::size_t hessian_power_iterations = 2;
  std::size_t hessian_cg_iterations = 30;
  std::size_t hessian_max_refreshes = 3;
  std::size_t hessian_factor_iterations = 5;
  double hessian_step = 1e-4;
  double transport_cov_shrink = 0.25;
  double transport_cov_ratio_cap = 4.0;
  double l = 0.0;
  std::size_t initial_transport_steps = 150;
  std::size_t transport_max_reflections = 500;
  double transport_initial_distance = 1.0;
  double transport_min_distance = 1e-8;
  double transport_max_distance = 1e6;
  double transport_max_logp_drop = 1000.0;
  double transport_max_segment_logp_drop = 500.0;
  double transport_max_endpoint_from_best_drop = 100.0;
  double transport_direction_persistence = 0.9;
  double transport_failure_direction_decay = 0.25;
  // Reflections the transport phase may spend, quoted per step but pooled
  // across the phase. transport_options_ multiplies it by
  // initial_transport_steps into a single ReflectedTransportOptions
  // ::reflection_budget, and each step then takes
  // max(1, min(transport_max_reflections, budget - used)). So a step is free
  // to reflect far past 75 as long as other steps spend less, and only the
  // phase total is bounded. That pooling is the point: the long excursions
  // that carry a badly initialised chain to the bulk are rare, and a strict
  // per-step cap would truncate exactly those. Zero drops the pooled bound and
  // leaves only transport_max_reflections per step.
  std::size_t transport_reflection_budget_per_step = 75;
};

class BaseKLHR {
public:
  static constexpr int kDiag = 0;
  static constexpr int kSketch = 1;
  static constexpr int kHess = 2;
  static constexpr int kComponents = 3;
  // The direction that was actually used came from no component's law, so no
  // component may be credited or charged for how it performed.
  static constexpr int kUnattributed = -1;

  // Why a Hessian refresh gave up. Fifteen distinct abort paths made it
  // impossible to tell whether the failures that cost HESS the earnings
  // comparison were indefinite curvature in conjugate gradients or
  // something else entirely.
  //
  // Diagnostic only -- nothing in the sampler branches on them; examples/
  // example.cpp reads them out through hessian_failures(). Keep them: the
  // cg_probe counter is what localised the conjugate-gradient breakdown that
  // was aborting every refresh on a low-dimensional target, and no aggregate
  // success rate could have told those apart.
  static constexpr int kFail_gated = 0;
  static constexpr int kFail_reference = 1;
  static constexpr int kFail_subspace = 2;
  static constexpr int kFail_cg_probe = 3;
  static constexpr int kFail_diag_ratio = 4;
  static constexpr int kFail_cg_sketch = 5;
  static constexpr int kFail_cg_power = 6;
  static constexpr int kFail_cg_final = 7;
  static constexpr int kFail_eigensolver = 8;
  static constexpr int kFail_factor_nonfinite = 9;
  static constexpr int kFail_degenerate_dir = 10;
  static constexpr int kFail_basis_nonfinite = 11;
  static constexpr int kFailCount = 12;

  std::size_t nfev_;
  double acceptance_rate_;
  double log_density_;

  BaseKLHR(std::string stan_file, std::string json_file,
           const KlhrOptions& options = KlhrOptions{}) :
    bsm_(stan_file, json_file),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(normalized_options_(options, bsm_.dim())),
    adaptation_steps_(post_transport_warmup_steps(
      opts_.warmup, opts_.initial_transport_steps)),
    adaptation_span_(adaptation_span_of_(adaptation_steps_,
                                         opts_.terminal_buffer)),
    transport_(bsm_.dim(), transport_options_(opts_)),
    windowed_adaptation_(adaptation_span_, opts_.windowsize,
                         opts_.windowscale),
    K_adaptation_(K_adaptation_config_(opts_, bsm_.dim())),
    online_moments_(bsm_.dim()),
    K_online_moments_(bsm_.dim()),
    online_pca_(bsm_.dim(), opts_.J, opts_.l, opts_.tol),
    sketch_(bsm_.dim(),
            opts_.mixture_direction ? opts_.sketch_columns : 0,
            opts_.tol, opts_.seed ^ 0x5bf03635U),
    projected_moments_(opts_.J) {

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
    theta_.resize(dim());
    theta_ = bsm_.param_initialize(bsrng);

    w_.resize(opts_.N);
    x_.resize(opts_.N);
    gauss_hermite(opts_.N, w_, x_);
    x_ *= std::sqrt(2.0);
    w_ /= std::sqrt(std::numbers::pi);

    initialize_pca_schedule_();
    reset_adaptation_to_defaults_(false);
    nfev_ = 0;
    acceptance_rate_ = 0.0;
    Eigen::VectorXd initial_grad = Eigen::VectorXd::Zero(dim());
    bsm_.log_density_gradient_noe(theta_, log_density_, initial_grad);
    ++nfev_;

    initialize_K_window_metric_(false);

    transport_.initialize({theta_, initial_grad, log_density_, true},
                          rng_, std_normal_);
    draw_ = 0;
  }

  virtual ~BaseKLHR() = default;

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  // The two names are not redundant: diagonal_variance_() says what the value
  // is (the sanitised diagonal of cov_), metric_variance() says what it is
  // for. Renaming the private one would lose the distinction between the
  // diagonal and the full metric, which the low-rank components also feed.
  Eigen::VectorXd metric_variance() const { return diagonal_variance_(); }
  Eigen::VectorXd transport_covariance() const { return transport_cov_; }

  bool hess_gate_open() const { return hess_gate_open_(); }
  // Two candidate gate signals native to the Hessian refresh, both available
  // from the first refresh onwards.
  double hess_anisotropy() const {
    return hess_signal_count_ == 0 ? 0.0 :
      hess_anisotropy_total_ / static_cast<double>(hess_signal_count_);
  }
  // Mean overlap between consecutive windows' sketch subspaces: 1 means the
  // same directions every time, rank/D means freshly random.
  double sketch_persistence() const {
    return sketch_persistence_count_ == 0 ? 0.0 :
      sketch_persistence_total_ / static_cast<double>(sketch_persistence_count_);
  }

  double mixture_cov_beta_used() const { return mixture_cov_beta_used_; }
  double cov_explained_raw() const { return cov_explained_raw_; }
  Eigen::Index sketch_rank() const { return sketch_rank_; }
  std::size_t sketch_dropouts() const { return sketch_dropouts_; }
  double sketch_mean_rank() const {
    return sketch_rebuilds_ == 0 ? 0.0 :
      static_cast<double>(sketch_rank_total_) /
      static_cast<double>(sketch_rebuilds_);
  }

  std::array<std::size_t, kFailCount> hessian_failures() const {
    return hess_fail_;
  }

  // Diagnostic, and live: examples/example.cpp labels the hessian_failures()
  // histogram with it. Both are behind `requires` checks there, so dropping
  // either silently deletes the output rather than failing the build.
  static const char* hessian_failure_name(const int i) {
    static const char* names[kFailCount] = {"gated", "reference", "subspace", "cg_probe", "diag_ratio", "cg_sketch", "cg_power", "cg_final", "eigensolver", "factor_nonfinite", "degenerate_dir", "basis_nonfinite"};
    return (i >= 0 && i < kFailCount) ? names[i] : "?";
  }


  std::size_t hessian_attempts() const { return hessian_attempts_; }
  std::size_t hessian_successes() const { return hessian_refreshes_; }


  // Final mixture weights, in component order: DIAG, SKETCH, HESS.
  //
  // To judge whether the mixture is worth its cost, read a weight against the
  // floor it cannot go below (mixture_floor_*) and against the prior it decays
  // to when rewards are uninformative (mixture_prior_*). A component parked at
  // its floor was tried and rejected by the bandit; a component at zero never
  // became live at all, which hessian_successes() and sketch_dropouts()
  // separate. The two call for opposite responses and no end-to-end error
  // metric distinguishes them.
  //
  // A weight well above its prior is the bandit reporting a real per-coordinate
  // ESJD gain, but that is its own objective, not the run's -- confirm it
  // against standardized RMSE or min-ESS on the model in question before
  // reading a large weight as a win.
  std::array<double, kComponents> mixture_weights() const {
    return mixture_p_;
  }

  std::size_t overrelaxation_K() const {
    return opts_.K;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    if (draw_ <= opts_.initial_transport_steps) {
      auto result = transport_.step(
        bsm_, rng_, std_uniform_, std_normal_);
      nfev_ += result.evaluations;
      theta_ = result.state.theta;
      log_density_ = result.state.log_density;
      if (draw_ == opts_.initial_transport_steps) {
        apply_transport_handoff_(transport_.finish(rng_, std_normal_));
      }
      return bsm_.param_constrain(theta_);
    }

    const std::size_t K_draw = post_transport_warmup_draw(
      draw_, opts_.warmup, opts_.initial_transport_steps);
    if (adapting_(K_draw) && K_adaptation_.enabled()) {
      opts_.K = K_adaptation_.K();
    }
    Eigen::VectorXd rho = random_direction();
    const KlStepDiagnostics diagnostics = kl_step_(rho);
    if (adapting_(K_draw)) {
      bandit_observe_(rho, diagnostics);
      if (K_adaptation_.enabled()) {
        adapt_K_warmup_(rho, diagnostics);
      }
    }
    adapt_warmup_(theta_, K_draw);
    return bsm_.param_constrain(theta_);
  }

  Eigen::VectorXd random_direction() {
    const Eigen::Index D = dim();
    // One direction model throughout: a Gaussian shaped like the current
    // covariance estimate, degrading to its diagonal when no low-rank basis
    // is available yet.
    const bool lowrank_allowed =
      opts_.lowrank_during_warmup || draw_ > opts_.warmup;
    Eigen::VectorXd rho;
    if (opts_.mixture_direction) {
      rho = mixture_direction_noise_();
    } else if (lowrank_allowed && lowrank_ready_) {
      rho = direction_noise_();
    } else {
      rho = diagonal_direction_noise_();
    }

    // Both salvage paths below replace the chosen component's draw with
    // something that is not its law -- isotropic noise, or a coordinate axis.
    // Crediting the resulting jump to the component that was chosen would feed
    // the bandit a reward it did not earn, and crediting it to DIAG is no
    // better, since neither is the diagonal law either. Leave it unattributed
    // and let bandit_observe_ drop the draw.
    double norm = rho.norm();
    if (!std::isfinite(norm) || norm <= opts_.tol) {
      last_component_ = kUnattributed;
      rho = normal_rng_(D);
      norm = rho.norm();
    }

    if (std::isfinite(norm) && norm > 0.0) {
      rho /= norm;
    } else {
      last_component_ = kUnattributed;
      rho = Eigen::VectorXd::Zero(D);
      rho(0) = 1.0;
    }

    return rho;
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
    bool hessian_usable = false;
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
    double inverse_hessian = std::numeric_limits<double>::quiet_NaN();
    if (mode.hess_inv.rows() == 1 && mode.hess_inv.cols() == 1) {
      inverse_hessian = mode.hess_inv(0, 0);
    }
    out.hessian_usable =
      std::isfinite(inverse_hessian) && inverse_hessian > 0.0;
    if (out.hessian_usable) {
      out.log_scale = 0.5 * std::log(inverse_hessian);
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

    if (!mode.success || !mode.hessian_usable) {
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

  static KlhrOptions normalized_options_(KlhrOptions options,
                                         const Eigen::Index dim) {
    const Eigen::Index D = dim;
    options.N = std::max<Eigen::Index>(1, options.N);
    if (!(options.tol > 0.0) || !std::isfinite(options.tol)) {
      options.tol = 1e-10;
    }
    if (std::isfinite(options.grad_clip)) {
      options.grad_clip = std::abs(options.grad_clip);
    } else {
      options.grad_clip = positive_infinity;
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
    options.lowrank_min_activations =
      std::max<std::size_t>(1, options.lowrank_min_activations);
    const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
    options.K = force_odd_K_(std::min(options.K, int_max));
    options.K_max = force_odd_K_(std::min(options.K_max, int_max));
    // The transport phase consumes warmup draws without adapting, so a
    // transport longer than warmup silently disables every adaptation.
    options.initial_transport_steps =
      std::min(options.initial_transport_steps, options.warmup);
    options.K_windowsize = std::max<std::size_t>(1, options.K_windowsize);
    options.K_windowscale = std::max<std::size_t>(1, options.K_windowscale);
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);
    if (!std::isfinite(options.l)) {
      options.l = 0.0;
    }
    options.J = std::clamp(options.J, Eigen::Index{0}, D);
    options.transport_J = std::clamp(options.transport_J, Eigen::Index{0}, D);
    // Probes beyond D buy nothing and cost more than the matrix they avoid.
    //
    // The Nystrom identity S ~ Y (Omega' S Omega)^+ Y' is exact once Omega
    // spans R^D, so for m >= D there is no approximation error left to reduce
    // and the extra columns are pure overhead.
    options.sketch_columns = std::clamp(options.sketch_columns,
                                        Eigen::Index{1},
                                        std::max<Eigen::Index>(D, 1));
    options.direction_lowrank_weight =
      std::isfinite(options.direction_lowrank_weight) ?
      std::clamp(options.direction_lowrank_weight, 0.0, 1.0) : 1.0;
    options.direction_min_diag_fraction =
      std::isfinite(options.direction_min_diag_fraction) ?
      std::clamp(options.direction_min_diag_fraction, 0.0, 1.0) : 0.1;
    options.pca_freeze_fraction =
      std::isfinite(options.pca_freeze_fraction) ?
      std::clamp(options.pca_freeze_fraction, 0.0, 1.0) : 0.1;
    options.transport_cov_shrink =
      std::isfinite(options.transport_cov_shrink) ?
      std::clamp(options.transport_cov_shrink, 0.0, 1.0) : 0.25;
    options.transport_cov_ratio_cap =
      std::isfinite(options.transport_cov_ratio_cap) ?
      std::max(options.transport_cov_ratio_cap, 1.0) : 4.0;
    if (std::isfinite(options.transport_max_endpoint_from_best_drop)) {
      options.transport_max_endpoint_from_best_drop =
        std::max(0.0, options.transport_max_endpoint_from_best_drop);
    } else {
      options.transport_max_endpoint_from_best_drop =
        positive_infinity;
    }
    return options;
  }

  static KAdaptationConfig K_adaptation_config_(
      const KlhrOptions& options,
      const Eigen::Index dim) {
    return {
      .enabled = options.adapt_K,
      .initial_K = options.K,
      .maximum_K = options.K_max,
      .warmup_steps = adaptation_span_of_(
        post_transport_warmup_steps(
          options.warmup, options.initial_transport_steps),
        options.terminal_buffer),
      .windowsize = options.K_windowsize,
      .windowscale = options.K_windowscale,
      .dimension = static_cast<std::size_t>(std::max<Eigen::Index>(1, dim)),
      .refresh_lags = options.K_refresh_lags,
      .interleave_trials = options.K_interleave_trials,
      .tolerance = options.tol,
    };
  }

  static ReflectedTransportOptions transport_options_(
      const KlhrOptions& options) {
    return {
      .N = options.N,
      .J = options.transport_J,
      .tol = options.tol,
      .grad_clip = options.grad_clip,
      .gtol = options.gtol,
      .maxiter_bfgs = options.transport_maxiter_bfgs,
      .pca_l = options.l,
      .covariance_shrink = options.transport_cov_shrink,
      .covariance_ratio_cap = options.transport_cov_ratio_cap,
      .max_reflections = options.transport_max_reflections,
      .initial_distance = options.transport_initial_distance,
      .min_distance = options.transport_min_distance,
      .max_distance = options.transport_max_distance,
      .max_logp_drop = options.transport_max_logp_drop,
      .max_segment_logp_drop = options.transport_max_segment_logp_drop,
      .max_endpoint_from_best_drop =
        options.transport_max_endpoint_from_best_drop,
      .direction_persistence = options.transport_direction_persistence,
      .failure_direction_decay = options.transport_failure_direction_decay,
      .reflection_budget =
        options.transport_reflection_budget_per_step == 0 ? 0 :
        options.transport_reflection_budget_per_step *
        options.initial_transport_steps,
    };
  }

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;

  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;

  KlhrOptions opts_;
  // Warmup draws that actually reach the adaptation, i.e. after transport.
  std::size_t adaptation_steps_;
  // Post-transport draws that actually adapt, i.e. excluding the terminal
  // buffer. Windows and the K ladder are laid out over this, so everything is
  // already frozen before the buffer begins.
  std::size_t adaptation_span_;
  ReflectedTransport transport_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  KWindowedAdaptation K_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;
  mcmcpp::WelfordAccumulator K_online_moments_;
  OnlinePCA online_pca_;
  Sketch sketch_;
  mcmcpp::WelfordAccumulator projected_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd x_; // Gauss-Hermite sample points
  Eigen::VectorXd w_; // and weights
  Eigen::VectorXd mean_;
  Eigen::VectorXd cov_;
  // sanitize_variance_(cov_), kept in step by set_covariance_.
  Eigen::VectorXd diagonal_variance_cache_;
  Eigen::VectorXd K_center_;
  Eigen::VectorXd K_variance_;
  Eigen::MatrixXd eigvecs_;
  Eigen::VectorXd eigvals_;
  Eigen::MatrixXd projection_basis_;
  std::array<std::size_t, kFailCount> hess_fail_{};
  Eigen::VectorXd transport_cov_;
  Eigen::MatrixXd sketch_prev_basis_;
  double sketch_persistence_total_ = 0.0;
  std::size_t sketch_persistence_count_ = 0;
  std::size_t hessian_refreshes_ = 0;
  std::size_t hessian_attempts_ = 0;
  // Mixture direction law and its bandit.
  Eigen::MatrixXd mixture_sketch_basis_;
  double mixture_cov_beta_used_ = 0.0;
  double cov_explained_raw_ = 0.0;
  Eigen::MatrixXd mixture_sketch_scaled_;
  Eigen::VectorXd mixture_sketch_residual_;
  Eigen::MatrixXd mixture_hess_basis_;
  bool mixture_hess_ready_ = false;
  bool mixture_sketch_ready_ = false;
  // Initialised to the prior by reset_mixture_(), which the constructor runs.
  std::array<double, kComponents> mixture_logw_{{0.0, 0.0, 0.0}};
  std::array<double, kComponents> mixture_p_{{1.0, 0.0, 0.0}};
  struct BanditBlock {
    std::array<Eigen::VectorXd, kComponents> sum;
    std::array<std::size_t, kComponents> cnt{{0, 0, 0}};
  };
  std::map<std::size_t, BanditBlock> blocks_;
  std::size_t bandit_updates_ = 0;
  int last_component_ = 0;
  // The mean the sketch is currently centering on; the window mean moves
  // away from it, and recenter() removes the resulting bias.
  Eigen::VectorXd sketch_center_;
  Eigen::Index sketch_rank_ = 0;
  std::size_t sketch_rebuilds_ = 0;
  std::size_t sketch_rank_total_ = 0;
  std::size_t sketch_dropouts_ = 0;
  double hess_anisotropy_total_ = 0.0;
  std::size_t hess_signal_count_ = 0;

  bool projection_basis_ready_ = false;
  bool pca_frozen_ = false;
  bool lowrank_ready_ = false;
  bool pca_calibration_enabled_ = false;
  bool K_radius_ready_ = false;
  std::size_t pca_freeze_draw_ = 0;
  std::size_t projected_pair_count_ = 0;

  std::size_t draw_;
  std::size_t kl_steps_ = 0;

  // Still needed: draw() hands this to bandit_observe_ and adapt_K_warmup_,
  // which between them read xi, acceptance_probability and valid. `accepted`
  // is the exception -- kl_step_ is its only reader, so it could be a local.
  // Left on the struct because a rejected step and an invalid one are
  // different things and having both fields visible keeps that legible at the
  // call sites.
  struct KlStepDiagnostics {
    double xi = 0.0;
    double acceptance_probability = 0.0;
    bool valid = false;
    bool accepted = false;
  };

  // Ordered overrelaxation is reversible with respect to the distribution it
  // is applied to (Neal 1995, section 4.2). Fitting one density per *line*
  // rather than one per endpoint therefore cancels the overrelaxation kernel
  // out of the Hastings ratio entirely, leaving the importance ratio
  // pi(t1)q(t0) / pi(t0)q(t1). That needs the fit to be a function of the
  // line alone, so the fit is centred on the projection of the adaptation
  // mean, which is the same point whichever end of the line we stand at.
  KlStepDiagnostics kl_step_(const Eigen::VectorXd& rho) {
    KlStepDiagnostics diagnostics;
    auto update_acceptance = [this](const bool accepted) {
      ++kl_steps_;
      const double d = accepted - acceptance_rate_;
      acceptance_rate_ += d / static_cast<double>(kl_steps_);
    };

    const double t0 = line_coordinate_(theta_, rho);
    if (!std::isfinite(t0)) {
      update_acceptance(false);
      return diagnostics;
    }
    const Eigen::VectorXd center = theta_ - t0 * rho;

    const Eigen::VectorXd eta = fit_line_(center, rho);
    const double t1 = overrelaxed_proposal_(eta, t0);
    const double xi = t1 - t0;
    diagnostics.xi = xi;
    if (!std::isfinite(xi)) {
      update_acceptance(false);
      return diagnostics;
    }
    const Eigen::VectorXd thetap = xi * rho + theta_;
    if (!thetap.allFinite()) {
      update_acceptance(false);
      return diagnostics;
    }

    const double ldp = bsm_.log_density_noe(thetap);
    ++nfev_;
    if (!std::isfinite(ldp)) {
      update_acceptance(false);
      return diagnostics;
    }

    const double log_q0 = log_line_density_(t0, eta);
    const double log_q1 = log_line_density_(t1, eta);
    if (!std::isfinite(log_q0) || !std::isfinite(log_q1)) {
      update_acceptance(false);
      return diagnostics;
    }

    const double a = ldp - log_density_ + log_q0 - log_q1;
    if (!std::isfinite(a)) {
      update_acceptance(false);
      return diagnostics;
    }
    diagnostics.valid = true;
    diagnostics.acceptance_probability =
      a >= 0.0 ? 1.0 : std::exp(a);
    const double log_u = std::log(std_uniform_(rng_));
    diagnostics.accepted = a >= 0.0 || log_u < a;
    update_acceptance(diagnostics.accepted);
    if (diagnostics.accepted) {
      theta_ = thetap;
      log_density_ = ldp;
    }
    return diagnostics;
  }

  // Position of theta along the line, measured from the projection of the
  // adaptation mean. Depends only on the line and on frozen adaptation
  // state, never on which endpoint is current.
  double line_coordinate_(const Eigen::VectorXd& theta,
                          const Eigen::VectorXd& rho) const {
    if (mean_.size() != theta.size()) {
      return theta.dot(rho);
    }
    return (theta - mean_).dot(rho);
  }

  // Floor relative to a typical variance, not absolutely. A coordinate that
  // happens not to move inside a window can report a variance many orders
  // below the rest; an absolute floor of tol lets it through, and everything
  // that divides by it -- the direction draw, the metric scale, the bandit
  // reward -- is then dominated by that one coordinate.
  Eigen::VectorXd sanitize_variance_(Eigen::VectorXd variance) const {
    std::vector<double> positive;
    positive.reserve(static_cast<std::size_t>(variance.size()));
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (std::isfinite(variance(d)) && variance(d) > 0.0) {
        positive.push_back(variance(d));
      }
    }
    double typical = 1.0;
    if (!positive.empty()) {
      const auto middle = positive.begin() + positive.size() / 2;
      std::nth_element(positive.begin(), middle, positive.end());
      typical = *middle;
    }
    const double floor = std::max(opts_.tol, 1e-6 * typical);
    for (Eigen::Index d = 0; d < variance.size(); ++d) {
      if (!std::isfinite(variance(d)) || variance(d) <= 0.0) {
        variance(d) = std::max(typical, floor);
      } else {
        variance(d) = std::max(variance(d), floor);
      }
    }
    return variance;
  }

  void initialize_K_window_metric_(const bool radius_ready) {
    K_center_ = theta_;
    // Already sanitised by set_covariance_.
    K_variance_ = diagonal_variance_();
    K_online_moments_.reset();
    K_radius_ready_ = radius_ready;
  }

  void update_K_window_metric_() {
    if (K_online_moments_.count() > 2) {
      Eigen::VectorXd center = K_online_moments_.mean();
      Eigen::VectorXd variance = K_online_moments_.variance();
      if (center.allFinite() && variance.allFinite()) {
        K_center_ = std::move(center);
        K_variance_ = sanitize_variance_(std::move(variance));
        K_radius_ready_ = true;
      }
    }
    K_online_moments_.reset();
  }

  double K_standardized_squared_jump_(
      const Eigen::VectorXd& rho,
      const double xi) const {
    if (!std::isfinite(xi) || K_variance_.size() != rho.size()) {
      return 0.0;
    }
    const double direction_precision =
      (rho.array().square() / K_variance_.array()).sum();
    const double jump = xi * xi * direction_precision /
      static_cast<double>(std::max<Eigen::Index>(1, dim()));
    return std::isfinite(jump) && jump >= 0.0 ? jump : 0.0;
  }

  double K_radius_() const {
    if (!K_radius_ready_ || K_center_.size() != theta_.size() ||
        K_variance_.size() != theta_.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd centered = theta_ - K_center_;
    const double radius_squared =
      (centered.array().square() / K_variance_.array()).sum();
    if (!std::isfinite(radius_squared) || radius_squared < 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double D = static_cast<double>(std::max<Eigen::Index>(1, dim()));
    return std::log1p(radius_squared / D);
  }

  void adapt_K_warmup_(const Eigen::VectorXd& rho,
                       const KlStepDiagnostics& diagnostics) {
    K_online_moments_.update(theta_);
    const double radius = K_radius_();
    const KAdaptationObservation observation = {
      .acceptance_probability = diagnostics.acceptance_probability,
      .standardized_squared_jump =
        K_standardized_squared_jump_(rho, diagnostics.xi),
      .valid = diagnostics.valid,
      .log_density = log_density_,
      .radius_available = K_radius_ready_ && std::isfinite(radius),
      .radius = radius,
    };
    const KAdaptationUpdate update = K_adaptation_.observe(observation);
    opts_.K = update.K;
    if (update.window_closed && !update.finalized) {
      update_K_window_metric_();
    }
  }

  void reset_adaptation_to_defaults_(const bool preserve_window_position) {
    const Eigen::Index D = dim();
    // Seed from the current draw rather than the origin: the mean anchors
    // the canonical line origin in kl_step_ and centres the PCA, and on a
    // badly scaled posterior the origin can be many standard deviations from
    // anywhere the chain will ever be.
    mean_ = (theta_.size() == D && theta_.allFinite()) ?
      theta_ : Eigen::VectorXd::Zero(D);
    set_covariance_(Eigen::VectorXd::Ones(D));
    eigvecs_ = Eigen::MatrixXd::Zero(D, opts_.J);
    eigvals_ = Eigen::VectorXd::Ones(opts_.J);
    projection_basis_ = Eigen::MatrixXd::Zero(D, opts_.J);
    hessian_refreshes_ = 0;
    hess_anisotropy_total_ = 0.0;
    hess_signal_count_ = 0;

    online_moments_.reset();
    reset_basis_estimator_();
    projected_moments_.reset();
    if (!preserve_window_position) {
      windowed_adaptation_.reset();
    }

    projection_basis_ready_ = false;
    pca_frozen_ = false;
    lowrank_ready_ = false;
    projected_pair_count_ = 0;
    reset_mixture_();
  }

  // cov_ moves only at a window close, at the transport handoff, and on a
  // reset, so the sanitised form is cached at those three points rather than
  // rebuilt on every read. The read side is hot: the direction draw and the
  // bandit reward each want it once per draw, and sanitize_variance_ costs an
  // allocation plus an nth_element over D.
  //
  // Route every write through here so the two cannot drift apart.
  void set_covariance_(Eigen::VectorXd covariance) {
    cov_ = std::move(covariance);
    diagonal_variance_cache_ = sanitize_variance_(cov_);
  }

  const Eigen::VectorXd& diagonal_variance_() const {
    return diagonal_variance_cache_;
  }

  Eigen::VectorXd metric_scale_() const {
    return diagonal_variance_().array().sqrt().matrix();
  }

  // Whitened Hessian applied to each column of V: scale .* (H (scale .* v)),
  // by central differencing of gradients. Two gradient evaluations per
  // column. Working whitened keeps the step dimensionless, which matters on
  // a posterior whose coordinates differ by orders of magnitude.
  bool whitened_hessian_apply_(const Eigen::VectorXd& reference,
                               const Eigen::VectorXd& scale,
                               const Eigen::MatrixXd& V,
                               Eigen::MatrixXd& out) {
    const double h = opts_.hessian_step;
    out.resize(V.rows(), V.cols());
    Eigen::VectorXd forward(dim());
    Eigen::VectorXd backward(dim());
    double value = 0.0;
    for (Eigen::Index j = 0; j < V.cols(); ++j) {
      const Eigen::VectorXd delta =
        h * (scale.array() * V.col(j).array()).matrix();
      bsm_.log_density_gradient_noe(reference + delta, value, forward);
      bsm_.log_density_gradient_noe(reference - delta, value, backward);
      nfev_ += 2;
      if (!forward.allFinite() || !backward.allFinite()) {
        return false;
      }
      // f = -log pi, so H v = -(grad log pi(+) - grad log pi(-)) / (2h).
      out.col(j) = (scale.array() *
                    (backward - forward).array() / (2.0 * h)).matrix();
    }
    return out.allFinite();
  }

  // Conjugate gradients on the whitened Hessian: returns Sigma_w B.
  // relative_residual, when given, receives the mean over columns of
  // ||r_j||^2 / ||b_j||^2 at the last iterate. This is the residual conjugate
  // gradients maintains recursively, which is what it costs nothing to report,
  // and it is the only visibility we have into whether hessian_cg_iterations
  // is enough on a given target.
  bool whitened_covariance_apply_(const Eigen::VectorXd& reference,
                                  const Eigen::VectorXd& scale,
                                  const Eigen::MatrixXd& B,
                                  Eigen::MatrixXd& X,
                                  double* relative_residual = nullptr) {
    X = Eigen::MatrixXd::Zero(B.rows(), B.cols());
    Eigen::MatrixXd R = B;
    Eigen::MatrixXd P = R;
    Eigen::MatrixXd AP;
    const Eigen::ArrayXd rs0 = B.colwise().squaredNorm().array();
    // Retire a column once its residual has collapsed relative to where it
    // started -- 1e-16 in squared norm, so 1e-8 in norm, the conventional
    // conjugate-gradient stopping point.
    //
    // Without this the solve cannot succeed on a low-dimensional target.
    // Conjugate gradients terminates in at most D steps, and past that P is
    // numerically zero: whitened_hessian_apply_ forms reference +/- h*scale*P,
    // which rounds back to reference exactly, so p'Ap comes back as zero or as
    // sign-random rounding noise and the indefiniteness test below reads it as
    // a failed refresh. Every D <= hessian_cg_iterations target aborts, which
    // is what left funnel and earnings with almost no successful refreshes.
    // Where nothing retires -- D well above hessian_cg_iterations -- the
    // arithmetic below is unchanged coefficient for coefficient.
    const Eigen::ArrayXd converged = rs0 * 1e-16;
    Eigen::ArrayXd active = Eigen::ArrayXd::Ones(B.cols());
    Eigen::ArrayXd rs = rs0;
    for (std::size_t iteration = 0;
         iteration < opts_.hessian_cg_iterations; ++iteration) {
      for (Eigen::Index j = 0; j < B.cols(); ++j) {
        if (active(j) > 0.0 && rs(j) <= converged(j)) {
          active(j) = 0.0;
          P.col(j).setZero();
          R.col(j).setZero();
        }
      }
      if ((active <= 0.0).all()) {
        break;
      }
      if (!whitened_hessian_apply_(reference, scale, P, AP)) {
        return false;
      }
      const Eigen::ArrayXd pap =
        (P.array() * AP.array()).colwise().sum().transpose();
      // A non-positive curvature means the operator is indefinite along this
      // search direction, so conjugate gradients cannot proceed. Declining is
      // the intended outcome. Retired columns carry p'Ap == 0 by construction,
      // so they are exempt.
      if (!pap.allFinite() || ((active > 0.0) && (pap <= 0.0)).any()) {
        return false;
      }
      // 1 - active keeps a retired column's 0/0 out of alpha. It multiplies a
      // zero P and a zero AP, so only finiteness matters.
      const Eigen::ArrayXd alpha = active * rs / (pap + (1.0 - active));
      X += P * alpha.matrix().asDiagonal();
      R -= AP * alpha.matrix().asDiagonal();
      const Eigen::ArrayXd rs_next = R.colwise().squaredNorm().array();
      const Eigen::ArrayXd beta = active * rs_next / rs.max(opts_.tol);
      P = R + P * beta.matrix().asDiagonal();
      rs = rs_next;
    }
    if (relative_residual != nullptr) {
      const Eigen::ArrayXd scale2 =
        B.colwise().squaredNorm().array().max(opts_.tol);
      const double mean = (rs / scale2).mean();
      *relative_residual = std::isfinite(mean) ? mean : 0.0;
    }
    return X.allFinite();
  }

  // A J-factor model on D variables has D + D*J - J(J-1)/2 free parameters
  // against the D(D+1)/2 distinct entries of Sigma, so it is only
  // identifiable while (D-J)^2 >= D+J. Beyond that the fit is chasing more
  // parameters than the data can pin down; the plug-in eigendecomposition is
  // the right estimator there, and at J = D it reconstructs Sigma exactly.
  //
  // The closed form is the Ledermann bound (Ledermann 1937; see also Anderson
  // and Rubin 1956). Requiring free parameters not to exceed distinct entries,
  //
  //   D + DJ - J(J-1)/2  <=  D(D+1)/2   <=>   (D-J)^2 >= D + J
  //                                     <=>   J^2 - (2D+1)J + D^2 - D >= 0,
  //
  // and the smaller root of that quadratic is
  //
  //   J <= [ (2D+1) - sqrt((2D+1)^2 - 4(D^2-D)) ] / 2
  //      = [ 2D + 1 - sqrt(8D + 1) ] / 2,
  //
  // which is what is returned, floored to an index. It is a necessary
  // condition on the parameter count, not a sufficient one for any particular
  // Sigma, so treat it as a ceiling rather than a guarantee.
  static Eigen::Index ledermann_bound_(const Eigen::Index D) {
    if (D <= 1) {
      return 0;
    }
    const double d = static_cast<double>(D);
    const double bound = 0.5 * (2.0 * d + 1.0 - std::sqrt(8.0 * d + 1.0));
    return std::max<Eigen::Index>(0, static_cast<Eigen::Index>(bound));
  }

  bool hessian_enabled_() const {
    return opts_.mixture_direction;
  }

  bool refresh_hessian_metric_() {
    if (!hessian_enabled_() || opts_.J <= 0 ||
        hessian_refreshes_ >= opts_.hessian_max_refreshes) {
      ++hess_fail_[kFail_gated];
      return false;
    }
    ++hessian_attempts_;
    return refresh_hessian_metric_attempt_();
  }

  // Only ever reached through refresh_hessian_metric_, which has already
  // charged the attempt and cleared the capability gate.
  bool refresh_hessian_metric_attempt_() {
    const Eigen::Index J = opts_.J;
    const Eigen::Index D = dim();
    const Eigen::VectorXd reference =
      (mean_.size() == D && mean_.allFinite()) ? mean_ : theta_;
    if (!reference.allFinite()) {
      ++hess_fail_[kFail_reference];
      return false;
    }
    const Eigen::VectorXd scale = metric_scale_();
    const Eigen::Index m = std::min(D, J + std::max<Eigen::Index>(
      0, opts_.hessian_oversample));
    if (m < J) {
      ++hess_fail_[kFail_subspace];
      return false;
    }

    const bool exact_diagonal = (m == D);
    Eigen::MatrixXd probe(D, m);
    if (exact_diagonal) {
      probe.setIdentity();
    } else {
      for (Eigen::Index j = 0; j < m; ++j) {
        probe.col(j) = normal_rng_(D);
      }
    }
    Eigen::MatrixXd probe_image;
    if (!whitened_covariance_apply_(reference, scale, probe, probe_image)) {
      ++hess_fail_[kFail_cg_probe];
      return false;
    }
    Eigen::VectorXd diagonal_ratio(D);
    if (exact_diagonal) {
      diagonal_ratio = probe_image.diagonal();
    } else {
      diagonal_ratio =
        (probe.array() * probe_image.array()).rowwise().mean();
    }
    if (!diagonal_ratio.allFinite()) {
      ++hess_fail_[kFail_diag_ratio];
      return false;
    }
    const Eigen::VectorXd psi_base = diagonal_ratio.cwiseMax(opts_.tol);

    Eigen::MatrixXd Y(D, m);
    for (Eigen::Index j = 0; j < m; ++j) {
      Y.col(j) = normal_rng_(D);
    }
    Eigen::MatrixXd sketch;
    if (!whitened_covariance_apply_(reference, scale, Y, sketch)) {
      ++hess_fail_[kFail_cg_sketch];
      return false;
    }
    for (std::size_t p = 0; p < opts_.hessian_power_iterations; ++p) {
      const Eigen::MatrixXd Q =
        Eigen::HouseholderQR<Eigen::MatrixXd>(sketch)
          .householderQ() * Eigen::MatrixXd::Identity(D, m);
      if (!whitened_covariance_apply_(reference, scale, Q, sketch)) {
        ++hess_fail_[kFail_cg_power];
        return false;
      }
    }
    const Eigen::MatrixXd Q =
      Eigen::HouseholderQR<Eigen::MatrixXd>(sketch)
        .householderQ() * Eigen::MatrixXd::Identity(D, m);
    Eigen::MatrixXd SQ;
    if (!whitened_covariance_apply_(reference, scale, Q, SQ)) {
      ++hess_fail_[kFail_cg_final];
      return false;
    }

    Eigen::MatrixXd small = Q.transpose() * SQ;
    small = 0.5 * (small + small.transpose()).eval();

    // Q' Sigma_w Q is the whitened covariance restricted to the sampled
    // subspace, from deliberate probes with power iterations. Recorded here
    // because it is available at the first successful refresh, which is
    // exactly when the Hessian component first becomes usable, so the gate
    // can never delay the component it governs.
    {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(small);
      if (spectrum.info() == Eigen::Success) {
        const Eigen::VectorXd& sv = spectrum.eigenvalues();
        const double median = sv(m / 2);
        if (std::isfinite(median) && median > opts_.tol) {
          hess_anisotropy_total_ += sv(m - 1) / median;
          ++hess_signal_count_;
        }
      }
    }

    // Factor analysis, fitted inside the sampled subspace: alternate the
    // loadings against the residual diagonal.
    //
    //   Sigma_w  ~  diag(psi) + V L V',   psi = 1 - diag(V L V')
    //
    // Plugging in the top eigenpairs of Sigma_w instead -- which is what
    // this did, and what the covariance path still does -- installs a
    // rank-J term whether or not there is any correlation to model. On an
    // exactly diagonal target Sigma_w is the identity, every eigenvalue is
    // one, the leading direction is arbitrary noise, and the metric picks
    // up a spurious off-diagonal worth roughly a factor of two in
    // cond(M^-1 Sigma). Fitting returns zero loadings there and the metric
    // collapses to the diagonal, which is the correct answer.
    //
    // Reusing Q costs nothing: the leading eigenvectors of Sigma_w - psi
    // are close to those of Sigma_w when psi is near isotropic, so no
    // further conjugate-gradient solves are needed.
    // Zero iterations reproduces the old plug-in exactly: psi stays zero, so
    // the single pass diagonalises Sigma_w itself.
    // Fit factors only where a factor model is identifiable; otherwise take
    // the plug-in eigendecomposition, which is exact once the subspace spans
    // the problem. On the four-parameter models the bound is one factor, and
    // fitting four cost earnings a factor of thirty in min-ESS.
    const bool fit_factors =
      opts_.hessian_factor_iterations > 0 && J <= ledermann_bound_(D);
    const std::size_t passes =
      std::max<std::size_t>(1, opts_.hessian_factor_iterations);
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(D);
    if (fit_factors) {
      psi = psi_base;
    }
    Eigen::MatrixXd whitened_basis = Eigen::MatrixXd::Zero(D, J);
    Eigen::VectorXd loadings = Eigen::VectorXd::Zero(J);
    for (std::size_t iteration = 0; iteration < passes; ++iteration) {
      Eigen::MatrixXd projected = small;
      projected.noalias() -= Q.transpose() * psi.asDiagonal() * Q;
      projected = (0.5 * (projected + projected.transpose())).eval();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(projected);
      if (solver.info() != Eigen::Success) {
        ++hess_fail_[kFail_eigensolver];
        return false;
      }
      for (Eigen::Index j = 0; j < J; ++j) {
        const Eigen::Index source = m - 1 - j;  // eigenvalues ascend
        // A negative loading would mean the direction is narrower than the
        // diagonal already says, which this model cannot represent.
        loadings(j) = std::max(solver.eigenvalues()(source), 0.0);
        whitened_basis.col(j) = Q * solver.eigenvectors().col(source);
      }
      if (fit_factors) {
        psi = psi_base -
          (whitened_basis.array().square().matrix() * loadings);
        psi = psi.cwiseMax(opts_.tol);
      }
      if (!psi.allFinite() || !loadings.allFinite()) {
        ++hess_fail_[kFail_factor_nonfinite];
        return false;
      }
    }

    Eigen::MatrixXd basis(D, J);
    Eigen::VectorXd weights(J);
    for (Eigen::Index j = 0; j < J; ++j) {
      // Sigma = diag(scale) Sigma_w diag(scale), so a whitened direction v
      // with loading l contributes l * (scale .* v)(scale .* v)'.
      const Eigen::VectorXd mapped =
        (scale.array() * whitened_basis.col(j).array()).matrix();
      const double norm = mapped.norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        // Only fatal if this direction was carrying a loading.
        if (loadings(j) > 0.0) {
          ++hess_fail_[kFail_degenerate_dir];
          return false;
        }
        basis.col(j) = Eigen::VectorXd::Unit(D, std::min<Eigen::Index>(j, D - 1));
        weights(j) = 0.0;
        continue;
      }
      basis.col(j) = mapped / norm;
      weights(j) = loadings(j) * norm * norm;
    }
    if (!basis.allFinite() || !weights.allFinite()) {
      ++hess_fail_[kFail_basis_nonfinite];
      return false;
    }

    eigvecs_.leftCols(J) = basis;
    eigvals_.head(J) = weights.cwiseMax(opts_.tol);
    lowrank_ready_ = true;
    ++hessian_refreshes_;
    return true;
  }

  static std::size_t adaptation_span_of_(const std::size_t steps,
                                         const std::size_t terminal) {
    // The buffer must never swallow the whole adaptation.
    if (terminal == 0 || steps <= terminal) {
      return steps;
    }
    return steps - terminal;
  }

  bool adapting_(const std::size_t adaptation_draw) const {
    return adaptation_draw > 0 && adaptation_draw <= adaptation_span_;
  }

  double mixture_floor_(const int c) const {
    if (c == kSketch) return opts_.mixture_floor_sketch;
    if (c == kHess) return opts_.mixture_floor_hess;
    return opts_.mixture_floor_diagonal;
  }

  double mixture_prior_(const int c) const {
    if (c == kSketch) return opts_.mixture_prior_sketch;
    if (c == kHess) return opts_.mixture_prior_hess;
    return 0.0;
  }

  // A component is live only once its block has actually been built from
  // data, which gives the "do not use an unestimated basis" gate for free.
  std::vector<int> live_components_() const {
    std::vector<int> live{kDiag};
    if (mixture_sketch_ready_) live.push_back(kSketch);
    if (mixture_hess_ready_) live.push_back(kHess);
    return live;
  }

  void refresh_mixture_weights_() {
    const std::vector<int> live = live_components_();
    double slack = 1.0;
    for (const int c : live) {
      slack -= mixture_floor_(c);
    }
    slack = std::max(slack, 0.0);

    double max_logw = -positive_infinity;
    for (const int c : live) {
      max_logw = std::max(max_logw, mixture_logw_[c]);
    }
    double total = 0.0;
    std::array<double, kComponents> w{{0.0, 0.0, 0.0}};
    for (const int c : live) {
      w[c] = std::exp(mixture_logw_[c] - max_logw);
      total += w[c];
    }
    mixture_p_ = {{0.0, 0.0, 0.0}};
    if (!(total > 0.0) || !std::isfinite(total)) {
      mixture_p_[kDiag] = 1.0;
      return;
    }
    for (const int c : live) {
      mixture_p_[c] = mixture_floor_(c) + slack * w[c] / total;
    }
  }

  // nu is symmetric and does not depend on theta, so nu(d)/nu(-d) is one and
  // the direction never enters the Hastings ratio -- which is what makes the
  // singular components legal despite having no Lebesgue density.
  Eigen::VectorXd mixture_direction_noise_() {
    const double u = std_uniform_(rng_);
    double cumulative = 0.0;
    int chosen = kDiag;
    for (int c = 0; c < kComponents; ++c) {
      cumulative += mixture_p_[c];
      if (u < cumulative) {
        chosen = c;
        break;
      }
    }

    if (chosen == kSketch && mixture_sketch_ready_) {
      last_component_ = kSketch;
      if (mixture_sketch_residual_.size() == dim()) {
        return scaled_additive_draw_(mixture_sketch_scaled_,
                                     mixture_sketch_residual_);
      }
      return mixture_sketch_basis_ *
        normal_rng_(mixture_sketch_basis_.cols());
    }
    if (chosen == kHess && mixture_hess_ready_) {
      last_component_ = kHess;
      // Pure low-rank, at the curvature's own magnitudes.
      return mixture_hess_basis_ * normal_rng_(mixture_hess_basis_.cols());
    }
    last_component_ = kDiag;
    return diagonal_direction_noise_();
  }

  // Rao-Blackwellised ESJD in the frozen diagonal metric: the acceptance
  // probability rather than the 0/1 outcome, which is the same expectation
  // with far less variance.
  void bandit_observe_(const Eigen::VectorXd& rho,
                       const KlStepDiagnostics& diagnostics) {
    if (!opts_.mixture_direction || !std::isfinite(diagnostics.xi) ||
        last_component_ < 0 || last_component_ >= kComponents) {
      return;
    }
    const Eigen::VectorXd& variance = diagonal_variance_();
    // per coordinate: alpha * (xi rho_d)^2 / sigma_d^2
    const Eigen::VectorXd contribution =
      (diagnostics.acceptance_probability * diagnostics.xi * diagnostics.xi) *
      (rho.array().square() / variance.array()).matrix();
    if (!contribution.allFinite() || (contribution.array() < 0.0).any()) {
      return;
    }
    // Stratify by overrelaxation level rather than restarting on every
    // change: K interleaves in short trial blocks, so resetting there left
    // every block below bandit_min_draws and the weights never moved at all.
    BanditBlock& block = blocks_[opts_.K];
    Eigen::VectorXd& target = block.sum[last_component_];
    if (target.size() != contribution.size()) {
      target = Eigen::VectorXd::Zero(contribution.size());
    }
    target += contribution;
    ++block.cnt[last_component_];
  }

  // Close a reward block and move the weights: a replicator step against the
  // mixture-average reward, plus shrinkage toward a diagonal-favouring prior
  // so an uninformative reward decays a component rather than leaving it
  // parked wherever it started. (The two halves are not in tension -- the
  // replicator term is the gradient step and the shrinkage term is the
  // regulariser on it; both appear in the single update at the end.)
  //
  // The mixture's per-coordinate ESJD is E_d(p) = sum_c p_c e_cd, and the
  // objective is Phi(p) = -sum_d 1/E_d -- a monotone transform of the
  // harmonic mean of the E_d, and a direct proxy for -sum_d tau_d, which is
  // what standardized RMSE measures. Phi is concave in p (its Hessian is
  // -2 sum_d e_cd e_bd / E_d^3, a negatively weighted Gram), so the
  // multiplicative update below is mirror ascent on a concave objective and
  // has a unique optimum on the simplex.
  //
  // Conveniently sum_c p_c g_c = sum_d E_d / E_d^2 = sum_d 1/E_d, so the
  // baseline the replicator already divides by is -Phi itself, and the
  // update keeps exactly the shape it had under the scalar reward.
  void finalize_bandit_block_() {
    if (!opts_.mixture_direction) {
      return;
    }
    const std::vector<int> live = live_components_();
    if (live.size() <= 1) {
      blocks_.clear();
      return;
    }

    std::array<double, kComponents> ratio_sum{{0.0, 0.0, 0.0}};
    std::size_t strata = 0;
    for (const auto& [level, block] : blocks_) {
      (void) level;
      bool enough = true;
      for (const int c : live) {
        if (block.cnt[c] < opts_.bandit_min_draws ||
            block.sum[c].size() == 0) {
          enough = false;
        }
      }
      if (!enough) {
        continue;
      }

      std::array<Eigen::VectorXd, kComponents> e;
      Eigen::VectorXd mixed;
      for (const int c : live) {
        e[c] = block.sum[c] / static_cast<double>(block.cnt[c]);
        if (mixed.size() == 0) {
          mixed = Eigen::VectorXd::Zero(e[c].size());
        }
        mixed += mixture_p_[c] * e[c];
      }
      if (mixed.size() == 0 || !mixed.allFinite()) {
        continue;
      }

      // The old claim here -- that only DIAG moves every coordinate almost
      // surely -- is wrong, and SKETCH is the counterexample: whenever
      // build_scaled_component_ succeeds, mixture_direction_noise_ takes the
      // scaled_additive_draw_ branch, which adds a full-rank residual term and
      // so moves every coordinate too. HESS is the only genuinely singular
      // component; it draws inside a J-dimensional subspace and leaves the
      // complement exactly untouched.
      //
      // So the floor that keeps every E_d positive is DIAG's *or* SKETCH's,
      // whichever is live -- and SKETCH is not always live. The guard below
      // therefore still earns its place: it covers the case where the sketch
      // has dropped out and HESS holds weight, and any coordinate no live
      // component happened to move within the block.
      // Guard against a coordinate no component happened to touch.
      const double floor = std::max(opts_.tol, 1e-12 * mixed.maxCoeff());
      const Eigen::ArrayXd E = mixed.array().max(floor);
      const Eigen::ArrayXd inv = 1.0 / E;
      const double baseline = inv.sum();
      std::array<double, kComponents> gradient{{0.0, 0.0, 0.0}};
      for (const int c : live) {
        gradient[c] = (e[c].array() * inv.square()).sum();
      }
      if (!std::isfinite(baseline) || baseline <= 0.0) {
        continue;
      }
      bool usable = true;
      for (const int c : live) {
        if (!std::isfinite(gradient[c])) {
          usable = false;
        }
      }
      if (!usable) {
        continue;
      }
      for (const int c : live) {
        ratio_sum[c] += gradient[c] / baseline;
      }
      ++strata;
    }

    const double gamma = opts_.bandit_gamma0 /
      std::pow(1.0 + static_cast<double>(bandit_updates_), opts_.bandit_decay);
    for (const int c : live) {
      // With no usable stratum the reward term is zero and only the prior
      // pulls, so an uninformative component decays toward the diagonal
      // instead of parking wherever it started.
      const double reward_term = strata > 0
        ? ratio_sum[c] / static_cast<double>(strata) - 1.0 : 0.0;
      mixture_logw_[c] += gamma * reward_term -
        gamma * opts_.bandit_shrink *
          (mixture_logw_[c] - mixture_prior_(c));
    }
    ++bandit_updates_;

    blocks_.clear();
    refresh_mixture_weights_();
  }

  void rebuild_mixture_() {
    if (!opts_.mixture_direction) {
      return;
    }

    // mean_ was updated at this window close, but the sketch accumulated
    // against the previous one; remove that bias before anything reads it.
    if (sketch_center_.size() == dim() && mean_.size() == dim()) {
      sketch_.recenter(mean_ - sketch_center_);
    }

    Eigen::MatrixXd basis;
    mixture_sketch_ready_ = sketch_.factor(basis, opts_.sketch_max_rank);
    ++sketch_rebuilds_;
    if (mixture_sketch_ready_) {
      mixture_sketch_basis_ = basis;
      sketch_rank_ = mixture_sketch_basis_.cols();
      (void) build_scaled_component_(mixture_sketch_basis_,
                                     mixture_sketch_scaled_,
                                     mixture_sketch_residual_);

      {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(mixture_sketch_basis_);
        const Eigen::MatrixXd Qn =
          qr.householderQ() * Eigen::MatrixXd::Identity(
            mixture_sketch_basis_.rows(), mixture_sketch_basis_.cols());
        if (sketch_prev_basis_.size() > 0 &&
            sketch_prev_basis_.rows() == Qn.rows()) {
          const double denom = static_cast<double>(
            std::min(sketch_prev_basis_.cols(), Qn.cols()));
          const double overlap =
            (sketch_prev_basis_.transpose() * Qn).squaredNorm() /
            std::max(denom, 1.0);
          if (std::isfinite(overlap)) {
            sketch_persistence_total_ += overlap;
            ++sketch_persistence_count_;
          }
        }
        sketch_prev_basis_ = Qn;
      }
    } else {
      sketch_rank_ = 0;
      ++sketch_dropouts_;
    }
    sketch_rank_total_ += static_cast<std::size_t>(sketch_rank_);

    // The Hessian metric owns eigvecs_/eigvals_; as a mixture component it
    // needs a square root, which is just the basis scaled by the root of its
    // weights. refresh_hessian_metric_ ran before this at the window close,
    // and it self-limits to hessian_max_refreshes, after which the basis is
    // simply held.
    if (hess_gate_open_() && hessian_refreshes_ > 0 && opts_.J > 0) {
      const Eigen::VectorXd w = eigvals_.head(opts_.J).cwiseMax(0.0);
      Eigen::MatrixXd hb = eigvecs_.leftCols(opts_.J);
      for (Eigen::Index j = 0; j < opts_.J; ++j) {
        hb.col(j) *= std::sqrt(w(j));
      }
      if (hb.allFinite()) {
        mixture_hess_basis_ = std::move(hb);
        mixture_hess_ready_ = true;
      }
    } else {
      mixture_hess_ready_ = false;
    }

    // The sketch is centered on the outgoing mean, which has just moved,
    // and whitened by a scale that moved with it.
    sketch_.reset();
    sketch_center_ = mean_;
    refresh_mixture_weights_();
  }

  // Pool both generators' pairs and fit diag(psi) + V V' to them.
  //
  // Runs at the window close, against the scale the window's draws were
  // whitened by -- not the scale that has just been recomputed, which the
  // pairs know nothing about.
  // Before the first reading the behaviour is set by
  // anisotropy_gate_fail_open; after it, the statistic decides.
  bool hess_gate_open_() const {
    if (!(opts_.hess_gate_native_threshold > 0.0)) {
      return true;
    }
    if (hess_signal_count_ == 0) {
      return opts_.anisotropy_gate_fail_open;
    }
    return hess_anisotropy() > opts_.hess_gate_native_threshold;
  }

  // Rescale a low-rank factor so its marginals match the adapted diagonal,
  // and set aside the complementary residual. Applies to any component that
  // supplies a D-by-r factor.
  //
  // Two effects, worth separating. The draw becomes full rank, so successive
  // draws are not confined to the same r-dimensional subspace -- that
  // confinement is what the lag curves exposed, a real gain at lag 1 decaying
  // to a deficit by lag 200. And the marginals become exactly cov whatever the
  // component's own eigenvalues say, so only its *directions* matter. For the
  // sketch that is unambiguously right, its magnitudes being noise at ESS/D
  // below one, and it is applied there and nowhere else. HESS deliberately
  // keeps its own magnitudes: a curvature metric's whole content is that some
  // directions are far wider than the diagonal believes, and imposing cov
  // discards exactly that -- measured at 40.7x the diagonal sampler's lag-1
  // displacement unscaled against 1.02x scaled on ar1.
  //
  // The residual needs no floor: it is (1-beta) cov identically where the
  // scaling is unclamped and larger where clamped, so it stays positive and
  // the marginals stay exact.
  bool build_scaled_component_(const Eigen::MatrixXd& basis,
                               Eigen::MatrixXd& scaled,
                               Eigen::VectorXd& residual) {
    if (basis.size() == 0 || basis.rows() != dim()) {
      residual.resize(0);
      return false;
    }
    const Eigen::VectorXd& var = diagonal_variance_();
    const Eigen::VectorXd bb = basis.array().square().rowwise().sum();
    // trace(BB') is the variance the subspace carries; var.sum() is the total.
    const double total = var.sum();
    cov_explained_raw_ = total > opts_.tol ? bb.sum() / total : 0.0;
    const double beta = total > opts_.tol
      ? std::clamp(cov_explained_raw_, 0.0, 1.0) : 0.0;
    mixture_cov_beta_used_ = beta;
    const double guard = std::max(opts_.tol, 1e-8 * bb.maxCoeff());

    Eigen::VectorXd scale(var.size());
    for (Eigen::Index d = 0; d < var.size(); ++d) {
      scale(d) = std::sqrt(beta * var(d) / std::max(bb(d), guard));
    }
    scaled = scale.asDiagonal() * basis;
    const Eigen::VectorXd contribution =
      scaled.array().square().rowwise().sum();
    residual = (var - contribution).cwiseMax(opts_.tol);
    if (!residual.allFinite() || !scaled.allFinite()) {
      residual.resize(0);
      return false;
    }
    return true;
  }

  Eigen::VectorXd scaled_additive_draw_(const Eigen::MatrixXd& scaled,
                                        const Eigen::VectorXd& residual) {
    Eigen::VectorXd d =
      residual.array().sqrt().matrix().cwiseProduct(normal_rng_(dim()));
    d += scaled * normal_rng_(scaled.cols());
    return d;
  }

  void reset_mixture_() {
    mixture_sketch_ready_ = false;
    mixture_hess_ready_ = false;
    mixture_sketch_residual_.resize(0);
    for (int c = 0; c < kComponents; ++c) {
      mixture_logw_[c] = mixture_prior_(c);
    }
    blocks_.clear();
    bandit_updates_ = 0;
    last_component_ = kDiag;
    sketch_center_ = mean_;
    sketch_prev_basis_.resize(0, 0);
    sketch_persistence_total_ = 0.0;
    sketch_persistence_count_ = 0;
    sketch_rank_ = 0;
    sketch_rebuilds_ = 0;
    sketch_rank_total_ = 0;
    sketch_dropouts_ = 0;
    refresh_mixture_weights_();
  }

  Eigen::VectorXd direction_noise_() {
    const Eigen::Index D = dim();
    const double alpha = opts_.direction_lowrank_weight;
    const double min_diag_fraction = opts_.direction_min_diag_fraction;
    const Eigen::VectorXd& base_var = diagonal_variance_();
    Eigen::VectorXd residual_var = base_var;
    const Eigen::Index rank = lowrank_ready_ ? opts_.J : 0;
    if (rank == 0 || alpha == 0.0) {
      return diagonal_direction_noise_(base_var);
    }
    const Eigen::MatrixXd lowrank_basis = eigvecs_.leftCols(rank);
    const Eigen::VectorXd lowrank_variance = eigvals_.head(rank);
    residual_var -= alpha *
      (lowrank_basis.array().square().matrix() * lowrank_variance);

    for (Eigen::Index d = 0; d < D; ++d) {
      const double floor = std::max(opts_.tol, min_diag_fraction * base_var(d));
      if (!std::isfinite(residual_var(d)) || residual_var(d) < floor) {
        residual_var(d) = floor;
      }
    }

    Eigen::VectorXd noise = diagonal_direction_noise_(residual_var);
    const Eigen::VectorXd lowrank_sd =
      (alpha * lowrank_variance.array()).sqrt().matrix();
    noise += lowrank_basis *
      lowrank_sd.cwiseProduct(normal_rng_(rank));

    if (!noise.allFinite()) {
      return normal_rng_(D);
    }
    return noise;
  }

  Eigen::VectorXd diagonal_direction_noise_() {
    return diagonal_direction_noise_(diagonal_variance_());
  }

  Eigen::VectorXd diagonal_direction_noise_(
      const Eigen::VectorXd& variance) {
    return variance.array().sqrt().matrix().cwiseProduct(
      normal_rng_(dim()));
  }

  void initialize_pca_schedule_() {
    std::size_t final_start = 0;
    const auto& closures = windowed_adaptation_.closures();
    if (adaptation_span_ > 0) {
      final_start = 1;
      if (closures.size() >= 2 && closures.back() == adaptation_span_) {
        final_start = closures[closures.size() - 2] + 1;
      }
    }

    const std::size_t final_length = final_start <= adaptation_span_ ?
      adaptation_span_ - final_start + 1 : 0;
    // Zero-valued PCA options intentionally disable low-rank calibration.
    pca_calibration_enabled_ =
      opts_.J > 0 && opts_.pca_freeze_fraction > 0.0 && final_length > 2;
    if (!pca_calibration_enabled_) {
      pca_freeze_draw_ = adaptation_span_;
      return;
    }

    const auto tail = static_cast<std::size_t>(
      std::ceil(opts_.pca_freeze_fraction * final_length));
    const std::size_t tail_length =
      std::clamp<std::size_t>(tail, 1, final_length);
    pca_freeze_draw_ =
      std::max(final_start, adaptation_span_ - tail_length);
  }

  bool set_projection_basis_from_estimator_() {
    if (online_pca_.count() < opts_.J) {
      return false;
    }
    return set_projection_basis_(online_pca_.vectors());
  }

  void update_basis_estimator_(const Eigen::VectorXd& centered) {
    online_pca_.update(centered);
  }

  // The center OnlinePCA's states are taken relative to; see
  void reset_basis_estimator_() {
    online_pca_.reset();
    sketch_.reset();
  }

  void reset_pca_estimator_() {
    online_pca_.reset();
  }

  bool set_projection_basis_(Eigen::MatrixXd basis) {
    if (opts_.J <= 0 || basis.cols() < opts_.J || !basis.allFinite()) {
      projection_basis_ready_ = false;
      return false;
    }
    basis = basis.leftCols(opts_.J).eval();
    for (Eigen::Index j = 0; j < basis.cols(); ++j) {
      const double norm = basis.col(j).norm();
      if (!std::isfinite(norm) || norm <= opts_.tol) {
        projection_basis_ready_ = false;
        return false;
      }
      basis.col(j) /= norm;
    }
    projection_basis_ = std::move(basis);
    projection_basis_ready_ = true;
    return true;
  }

  void update_projected_moments_(const Eigen::VectorXd& theta) {
    if (!projection_basis_ready_) {
      return;
    }

    const Eigen::VectorXd projected = projection_basis_.transpose() * theta;
    if (projected.allFinite()) {
      projected_moments_.update(projected);
    }
  }

  void activate_projected_pair_() {
    if (!projection_basis_ready_ || projected_moments_.count() <= 2) {
      return;
    }

    Eigen::VectorXd variances = projected_moments_.variance();
    for (Eigen::Index j = 0; j < opts_.J; ++j) {
      if (!std::isfinite(variances(j)) || variances(j) <= opts_.tol) {
        variances(j) = opts_.tol;
      }
    }

    eigvecs_.leftCols(opts_.J) = projection_basis_;
    eigvals_.head(opts_.J) = variances;
    ++projected_pair_count_;
    lowrank_ready_ = projected_pair_count_ >= opts_.lowrank_min_activations;
  }

  void freeze_pca_for_final_calibration_() {
    if (pca_frozen_) {
      return;
    }

    // Bank the variances accumulated against the outgoing basis before
    // installing a new one, otherwise a whole window's worth of projected
    // moments is discarded in favour of the short calibration tail.
    activate_projected_pair_();

    const bool frozen = set_projection_basis_from_estimator_();
    if (!frozen && projected_pair_count_ > 0) {
      projection_basis_ = eigvecs_.leftCols(opts_.J);
      projection_basis_ready_ = projection_basis_.allFinite();
    }

    projected_moments_.reset();
    reset_pca_estimator_();
    pca_frozen_ = true;
  }

  // Seeding the mean matters twice over: it centres the first post-transport
  // PCA window, and it anchors the canonical line origin used by kl_step_.
  // It has to happen on the rollback path too, where the adaptation state is
  // otherwise reset to its defaults.
  void seed_mean_from_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    if (handoff.mean.size() == dim() && handoff.mean.allFinite()) {
      mean_ = handoff.mean;
    } else if (handoff.state.theta.size() == dim() &&
               handoff.state.theta.allFinite()) {
      mean_ = handoff.state.theta;
    }
  }

  void apply_transport_handoff_(
      const ReflectedTransport::Handoff& handoff) {
    theta_ = handoff.state.theta;
    log_density_ = handoff.state.log_density;
    if (handoff.rollback) {
      reset_adaptation_to_defaults_(true);
      seed_mean_from_handoff_(handoff);
      // The sketch accumulates theta - mean_, so its centre has to follow the
      // seeded mean. reset_mixture_() ran inside the reset above, before the
      // mean was seeded, and left it pointing at the pre-handoff value.
      sketch_center_ = mean_;
      initialize_K_window_metric_(false);
      return;
    }

    if (handoff.covariance.allFinite()) {
      set_covariance_(handoff.covariance);
      // Diagnostic only: the sampler reads cov_, never transport_cov_. It is
      // kept so the handoff estimate can be compared against the diagonal the
      // windows go on to produce, which examples/example.cpp prints beside
      // metric_variance(). Nothing else would notice its removal, and the
      // `requires` check there means the build would not either.
      transport_cov_ = handoff.covariance;
    }
    seed_mean_from_handoff_(handoff);
    // The transport's principal directions seed the low-rank basis, so the
    // projected moments start accumulating against it from the first
    // post-transport draw.
    if (!handoff.pca_ready || !set_projection_basis_(handoff.pca_basis)) {
      projection_basis_ready_ = false;
    }
    reset_basis_estimator_();
    // The sketch restarts here against the handoff mean. Leaving its centre at
    // the construction-time mean would make the first window's recenter()
    // subtract the wrong rank-one term -- on a badly scaled posterior, one
    // built from the distance between the initial point and the transport's
    // landing site, which can dwarf the window's own scatter.
    sketch_center_ = mean_;
    initialize_K_window_metric_(false);
  }

  // adaptation_draw counts from the end of the transport phase and is zero
  // outside post-transport warmup.
  void adapt_warmup_(const Eigen::VectorXd& theta,
                     const std::size_t adaptation_draw) {
    // Past the adapting span lies the terminal buffer, over which every slow
    // and fast parameter stays frozen so the chain equilibrates to the kernel
    // that will actually sample.
    if (!adapting_(adaptation_draw)) {
      return;
    }

    if (pca_calibration_enabled_ && !pca_frozen_ &&
        adaptation_draw >= pca_freeze_draw_) {
      freeze_pca_for_final_calibration_();
    }

    online_moments_.update(theta);
    if (opts_.mixture_direction) {
      // Centered, not whitened, and fed regardless of the rank-J calibration
      // schedule, which it does not participate in.
      //
      // An earlier comment here claimed these states were whitened, on the
      // argument that a raw sketch on a badly scaled target is dominated by
      // the largest-variance coordinates -- information the diagonal component
      // already holds -- so whitening is what leaves only correlation behind.
      // The argument was for a spectral shrinkage step that has since been
      // measured not to work and removed (see Sketch::factor). What replaced
      // it does the same job downstream instead: build_scaled_component_
      // rescales the basis so its marginals match the adapted diagonal, which
      // discards the sketch's own magnitudes and keeps only its directions.
      //
      // Feeding centered states is also what makes cov_explained_raw_ =
      // trace(BB')/trace(cov) a meaningful ratio bounded by one. Whiten here
      // and that reads as a rank rather than a fraction.
      sketch_.update(theta - mean_);
    } else if (pca_calibration_enabled_) {
      update_projected_moments_(theta);
      if (!pca_frozen_) {
        update_basis_estimator_(theta - mean_);
      }
    }

    if (windowed_adaptation_.window_closed(adaptation_draw)) {
      mean_ = online_moments_.mean();
      set_covariance_(online_moments_.variance());
      online_moments_.reset();

      // Refresh the Hessian metric against the updated diagonal.
      (void) refresh_hessian_metric_();
      // Slow parameters change only here; the bandit's block is closed too,
      // since its rewards were measured against the outgoing diagonal.
      if (opts_.mixture_direction) {
        finalize_bandit_block_();
        rebuild_mixture_();
      }

      if (pca_calibration_enabled_ && !pca_frozen_) {
        activate_projected_pair_();
        const bool has_next_basis = set_projection_basis_from_estimator_();
        projected_moments_.reset();
        if (!has_next_basis) {
          projection_basis_ready_ = false;
        }
        reset_pca_estimator_();
      }
    }

    if (pca_calibration_enabled_ && pca_frozen_ &&
        adaptation_draw == adaptation_span_) {
      activate_projected_pair_();
    }
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
      // q(.), while the reverse move from an unsaturated t1 would overrelax --
      // and proposing `from` itself instead was tried on exactly that
      // reasoning. It is worse on both counts. The imbalance is not repaired:
      // the identity sends zero mass out of the saturated set while
      // overrelaxation still sends mass in, so the set becomes absorbing, and
      // the flux mismatch stays O(q(saturated)) either way -- about 1e-16,
      // since the set is |z| > 8.15 under q. What the identity does change is
      // that a chain reaching that set can no longer leave, and on earnings
      // that turned a 1/40 failure-to-reach-the-posterior rate into 3/40:
      // seeds 2 (both samplers) and 18 froze near the initialisation point for
      // all 15000 sampling draws. The independence draw is the escape hatch,
      // and it is what makes the saturated branch a rescue rather than a trap.
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

    const int K = opts_.K;
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

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }

};

} // namespace klhr
