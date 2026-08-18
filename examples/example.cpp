#include "barker.hpp"
#include "mala.hpp"
#include "normal_klhr.hpp"
#include "sas_klhr.hpp"
#include "slice.hpp"
#include "stan.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char** argv) {

  // defaults
  std::uint64_t seed = 0;
  std::size_t num_warmup = 15'000;
  std::size_t num_iterations = 30'000;
  std::string model_name = "earnings";
  std::string sampler = "sas";
  std::string output = "draws/experiments.h5";
  Eigen::Index J = klhr::KlhrOptions{}.J;
  double target_accept = 0.8;
  double direction_lowrank_weight = klhr::KlhrOptions{}.direction_lowrank_weight;
  double direction_min_diag_fraction =
    klhr::KlhrOptions{}.direction_min_diag_fraction;
  bool lowrank_during_warmup = klhr::KlhrOptions{}.lowrank_during_warmup;
  double pca_freeze_fraction = klhr::KlhrOptions{}.pca_freeze_fraction;
  std::size_t lowrank_min_activations =
    klhr::KlhrOptions{}.lowrank_min_activations;
  Eigen::Index sketch_columns = klhr::KlhrOptions{}.sketch_columns;
  Eigen::Index transport_J = klhr::KlhrOptions{}.transport_J;
  bool mixture_direction = klhr::KlhrOptions{}.mixture_direction;
  bool position_sketch_direction =
    klhr::KlhrOptions{}.position_sketch_direction;
  bool curvature_sketch_direction = klhr::KlhrOptions{}.curvature_sketch_direction;
  Eigen::Index sketch_max_rank = klhr::KlhrOptions{}.sketch_max_rank;
  Eigen::Index curvature_max_rank = klhr::KlhrOptions{}.curvature_max_rank;
  double sketch_eigenvalue_cutoff =
    klhr::KlhrOptions{}.sketch_eigenvalue_cutoff;
  double sketch_residual_fraction =
    klhr::KlhrOptions{}.sketch_residual_fraction;
  double mixture_floor_diagonal = klhr::KlhrOptions{}.mixture_floor_diagonal;
  double mixture_floor_position = klhr::KlhrOptions{}.mixture_floor_position;
  double mixture_floor_curvature = klhr::KlhrOptions{}.mixture_floor_curvature;
  double bandit_shrink = klhr::KlhrOptions{}.bandit_shrink;
  double mixture_prior_position = klhr::KlhrOptions{}.mixture_prior_position;
  double mixture_prior_curvature = klhr::KlhrOptions{}.mixture_prior_curvature;
  std::size_t terminal_buffer = klhr::KlhrOptions{}.terminal_buffer;
  std::size_t initial_transport_steps = 150;
  std::size_t transport_max_reflections = 500;
  double transport_initial_distance = 1.0;
  double transport_min_distance = 1e-8;
  double transport_max_distance = 1e6;
  double transport_max_logp_drop = 1000.0;
  double transport_max_segment_logp_drop =
    klhr::KlhrOptions{}.transport_max_segment_logp_drop;
  double transport_max_endpoint_from_best_drop =
    klhr::KlhrOptions{}.transport_max_endpoint_from_best_drop;
  double transport_direction_persistence = 0.9;
  double transport_failure_direction_decay = 0.25;
  std::size_t transport_reflection_budget_per_step =
    klhr::KlhrOptions{}.transport_reflection_budget_per_step;
  double laplace_kl_residual_tol =
    klhr::KlhrOptions{}.laplace_kl_residual_tol;
  std::size_t K = klhr::KlhrOptions{}.K;
  bool adapt_K = klhr::KlhrOptions{}.adapt_K;
  std::size_t K_max = klhr::KlhrOptions{}.K_max;
  std::size_t K_refresh_lags = klhr::KlhrOptions{}.K_refresh_lags;
  bool K_interleave_trials = klhr::KlhrOptions{}.K_interleave_trials;

  {
    CLI::App app{"Run an MCMC sampler."};

    app.add_option("--seed", seed,
                   "Random seed (default 0 => random)")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--warmup", num_warmup,
                   "Number of warmup iterations")
      ->default_val(num_warmup)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--iterations", num_iterations,
                   "Number of total iterations (warmup included)")
      ->default_val(num_iterations)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--model", model_name,
                   "Stan model name");

    app.add_option("--sampler", sampler,
                   "Sampling algorithm: sas, normal, slice, barker, mala, or stan")
      ->check(CLI::IsMember(
        {"sas", "normal", "slice", "barker", "mala", "stan"}));

    app.add_option("--output", output,
                   "Path of the HDF5 file to write (truncated on open)")
      ->default_val(output);

    app.add_option("--J", J,
                   "Number of PCA directions to learn and use")
      ->default_val(J)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--target_accept", target_accept)
      ->default_val(0.8)
      ->check(CLI::PositiveNumber);

    app.add_option("--direction-lowrank-weight", direction_lowrank_weight,
                   "Weight for learned low-rank covariance in regular direction noise")
      ->default_val(direction_lowrank_weight);

    app.add_option("--direction-min-diag-fraction", direction_min_diag_fraction,
                   "Minimum retained fraction of componentwise variance in direction noise")
      ->default_val(direction_min_diag_fraction);

    app.add_flag("--lowrank-during-warmup,!--no-lowrank-during-warmup",
                 lowrank_during_warmup,
                 "Use calibrated low-rank direction noise during warmup once available")
      ->default_val(lowrank_during_warmup);

    app.add_option("--pca-freeze-fraction", pca_freeze_fraction,
                   "Fraction of the final adaptation window used to calibrate projected variances")
      ->default_val(pca_freeze_fraction);

    app.add_option("--lowrank-min-activations", lowrank_min_activations,
                   "Window closures required before the low-rank direction is used")
      ->default_val(lowrank_min_activations)
      ->check(CLI::PositiveNumber);

    app.add_option("--sketch-columns", sketch_columns,
                   "Random probes backing the mixture sketch component")
      ->default_val(sketch_columns)->check(CLI::PositiveNumber);

    app.add_option("--transport-J", transport_J,
                   "Rank of the reflected transport's own PCA (pinned, so "
                   "sweeping --J moves only the direction law)")
      ->default_val(transport_J)->check(CLI::NonNegativeNumber);

    app.add_flag("--mixture-direction,!--no-mixture-direction",
                 mixture_direction,
                 "Draw directions from an adaptively weighted mixture of "
                 "diagonal, position-sketch and direct-curvature components")
      ->default_val(mixture_direction);

    app.add_flag("--position-sketch,!--no-position-sketch",
                 position_sketch_direction,
                 "Enable the position-sketch mixture component")
      ->default_val(position_sketch_direction);

    app.add_flag("--curvature-sketch,!--no-curvature-sketch",
                 curvature_sketch_direction,
                 "Enable the direct-Hessian curvature mixture component")
      ->default_val(curvature_sketch_direction);

    app.add_option("--sketch-max-rank", sketch_max_rank,
                   "Maximum retained rank of the position sketch")
      ->default_val(sketch_max_rank)->check(CLI::PositiveNumber);

    app.add_option("--curvature-max-rank", curvature_max_rank,
                   "Maximum retained rank of the curvature component")
      ->default_val(curvature_max_rank)->check(CLI::PositiveNumber);

    app.add_option("--sketch-eigenvalue-cutoff", sketch_eigenvalue_cutoff,
                   "Retain standardized spectral values at least this large")
      ->default_val(sketch_eigenvalue_cutoff)->check(CLI::PositiveNumber);

    app.add_option("--sketch-residual-fraction", sketch_residual_fraction,
                   "Fraction of position-sketch variance left in its residual")
      ->default_val(sketch_residual_fraction)
      ->check(CLI::Range(0.0, 1.0));

    app.add_option("--mixture-floor-diagonal", mixture_floor_diagonal,
                   "Structural floor on the diagonal component's weight")
      ->default_val(mixture_floor_diagonal);

    app.add_option("--mixture-floor-position", mixture_floor_position,
                   "Structural floor on the position-sketch weight")
      ->default_val(mixture_floor_position);

    app.add_option("--mixture-floor-curvature", mixture_floor_curvature,
                   "Structural floor on the curvature-sketch weight")
      ->default_val(mixture_floor_curvature);

    app.add_option("--bandit-shrink", bandit_shrink,
                   "Shrinkage of the mixture weights toward their prior when "
                   "rewards are uninformative")
      ->default_val(bandit_shrink)->check(CLI::NonNegativeNumber);

    app.add_option("--mixture-prior-position", mixture_prior_position,
                   "Log-odds prior for the position-sketch component")
      ->default_val(mixture_prior_position);

    app.add_option("--mixture-prior-curvature", mixture_prior_curvature,
                   "Log-odds prior for the curvature-sketch component")
      ->default_val(mixture_prior_curvature);

    app.add_option("--terminal-buffer", terminal_buffer,
                   "Final warmup draws over which all adaptation is frozen")
      ->default_val(terminal_buffer)->check(CLI::NonNegativeNumber);

    app.add_option("--initial-transport-steps", initial_transport_steps,
                   "Initial nonstationary reflected-ray transport iterations")
      ->default_val(initial_transport_steps)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--transport-max-reflections", transport_max_reflections,
                   "Maximum specular reflection segments per initial transport iteration")
      ->default_val(transport_max_reflections)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--transport-initial-distance", transport_initial_distance,
                   "Initial Weibull scale guess for reflected transport")
      ->default_val(transport_initial_distance);

    app.add_option("--transport-min-distance", transport_min_distance,
                   "Minimum reflected transport ray distance")
      ->default_val(transport_min_distance);

    app.add_option("--transport-max-distance", transport_max_distance,
                   "Maximum reflected transport ray distance")
      ->default_val(transport_max_distance);

    app.add_option("--transport-max-logp-drop", transport_max_logp_drop,
                   "Maximum allowed log-density drop during reflected transport")
      ->default_val(transport_max_logp_drop);

    app.add_option("--transport-max-segment-logp-drop",
                   transport_max_segment_logp_drop,
                   "Maximum allowed log-density drop for one reflected transport segment")
      ->default_val(transport_max_segment_logp_drop);

    app.add_option("--transport-max-endpoint-from-best-drop",
                   transport_max_endpoint_from_best_drop,
                   "Maximum allowed final transport log-density drop from best transport state")
      ->default_val(transport_max_endpoint_from_best_drop);

    app.add_option("--transport-direction-persistence",
                   transport_direction_persistence,
                   "Partial direction refresh persistence for initial transport")
      ->default_val(transport_direction_persistence);

    app.add_option("--transport-failure-direction-decay",
                   transport_failure_direction_decay,
                   "Direction flip/damping factor after failed initial transport")
      ->default_val(transport_failure_direction_decay);

    app.add_option("--transport-reflection-budget-per-step",
                   transport_reflection_budget_per_step,
                   "Phase-wide reflection budget per transport step (0 = unlimited)")
      ->default_val(transport_reflection_budget_per_step)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--laplace-kl-residual-tol", laplace_kl_residual_tol,
                   "Accept the Laplace line fit when its KL gradient residual is below this")
      ->default_val(laplace_kl_residual_tol)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--K", K,
                   "Ordered overrelaxation level (rounded down to odd)")
      ->default_val(K)
      ->check(CLI::NonNegativeNumber);

    app.add_flag("--adapt-K,!--no-adapt-K", adapt_K,
                 "Adapt K during post-transport warmup")
      ->default_val(adapt_K);

    app.add_option("--K-refresh-lags", K_refresh_lags,
                   "Deepest lag the K refresh guard inspects (2 = original test)")
      ->default_val(K_refresh_lags)
      ->check(CLI::NonNegativeNumber);

    app.add_flag("--K-interleave,!--no-K-interleave", K_interleave_trials,
                 "Interleave K trial blocks instead of one contiguous run each")
      ->default_val(K_interleave_trials);

    app.add_option("--K-max", K_max,
                   "Largest K the adaptation ladder may reach")
      ->default_val(K_max)
      ->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions klhr_options = {
    .seed = seed,
    .laplace_kl_residual_tol = laplace_kl_residual_tol,
    .K = K,
    .adapt_K = adapt_K,
    .K_max = K_max,
    .K_refresh_lags = K_refresh_lags,
    .K_interleave_trials = K_interleave_trials,
    .warmup = num_warmup,
    .J = J,
    .direction_lowrank_weight = direction_lowrank_weight,
    .direction_min_diag_fraction = direction_min_diag_fraction,
    .lowrank_during_warmup = lowrank_during_warmup,
    .pca_freeze_fraction = pca_freeze_fraction,
    .lowrank_min_activations = lowrank_min_activations,
    .sketch_columns = sketch_columns,
    .transport_J = transport_J,
    .mixture_direction = mixture_direction,
    .position_sketch_direction = position_sketch_direction,
    .curvature_sketch_direction = curvature_sketch_direction,
    .sketch_max_rank = sketch_max_rank,
    .curvature_max_rank = curvature_max_rank,
    .sketch_eigenvalue_cutoff = sketch_eigenvalue_cutoff,
    .sketch_residual_fraction = sketch_residual_fraction,
    .mixture_floor_diagonal = mixture_floor_diagonal,
    .mixture_floor_position = mixture_floor_position,
    .mixture_floor_curvature = mixture_floor_curvature,
    .mixture_prior_position = mixture_prior_position,
    .mixture_prior_curvature = mixture_prior_curvature,
    .bandit_shrink = bandit_shrink,
    .terminal_buffer = terminal_buffer,
    .initial_transport_steps = initial_transport_steps,
    .transport_max_reflections = transport_max_reflections,
    .transport_initial_distance = transport_initial_distance,
    .transport_min_distance = transport_min_distance,
    .transport_max_distance = transport_max_distance,
    .transport_max_logp_drop = transport_max_logp_drop,
    .transport_max_segment_logp_drop = transport_max_segment_logp_drop,
    .transport_max_endpoint_from_best_drop =
      transport_max_endpoint_from_best_drop,
    .transport_direction_persistence = transport_direction_persistence,
    .transport_failure_direction_decay = transport_failure_direction_decay,
    .transport_reflection_budget_per_step =
      transport_reflection_budget_per_step,
  };

  auto run_sampler = [&](auto& algo) {

    Eigen::Index D = algo.dim();
    Eigen::MatrixXd draws(num_iterations, D);
    Eigen::VectorXd acceptance_rate(num_iterations);
    Eigen::VectorXd log_density(num_iterations);
    Eigen::VectorXd nfev(num_iterations);
    Eigen::VectorXd accept_stat;
    Eigen::VectorXd divergent;
    Eigen::VectorXd n_leapfrog;
    Eigen::VectorXd tree_depth;
    Eigen::VectorXd energy;
    Eigen::VectorXd stepsize;

    if constexpr (requires {
      algo.accept_stat();
      algo.divergent();
      algo.n_leapfrog();
      algo.tree_depth();
      algo.energy();
      algo.stepsize();
      algo.variance();
    }) {
      accept_stat.resize(num_iterations);
      divergent.resize(num_iterations);
      n_leapfrog.resize(num_iterations);
      tree_depth.resize(num_iterations);
      energy.resize(num_iterations);
      stepsize.resize(num_iterations);
    }

    Eigen::VectorXd draw(D);

    mcmcpp::WelfordAccumulator w{algo.dim()};
    mcmcpp::WelfordAccumulator msjd{};
    for (std::size_t n = 0; n < num_iterations; ++n) {
      draw = algo.draw();
      draws.row(n) = draw;
      acceptance_rate(n) = algo.acceptance_rate_;
      log_density(n) = algo.log_density_;
      nfev(n) = algo.nfev_;
      if constexpr (requires {
        algo.accept_stat();
        algo.divergent();
        algo.n_leapfrog();
        algo.tree_depth();
        algo.energy();
        algo.stepsize();
        algo.variance();
      }) {
        accept_stat(n) = algo.accept_stat();
        divergent(n) = algo.divergent();
        n_leapfrog(n) = algo.n_leapfrog();
        tree_depth(n) = algo.tree_depth();
        energy(n) = algo.energy();
        stepsize(n) = algo.stepsize();
      }
      if (n >= num_warmup) {
        w.update(draw);
        msjd.update((draws.row(n) - draws.row(n-1)).norm());
      }
    }

    std::cout << "Seed: " << algo.seed() << '\n';
    std::cout << "mean: " << w.mean().transpose() << '\n';
    std::cout << "std: " << w.std().transpose() << '\n';
    std::cout << "msjd: " << msjd.mean().transpose() << '\n';
    std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
    std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';
    if constexpr (requires { algo.metric_variance(); }) {
      const Eigen::VectorXd mv = algo.metric_variance();
      const Eigen::VectorXd tv = algo.transport_covariance();
      if (mv.size() <= 1200) {
        std::cout << "Adapted diagonal: " << mv.transpose() << '\n';
        if (tv.size() == mv.size()) {
          std::cout << "Transport handoff: " << tv.transpose() << '\n';
        }
      }
    }
    if constexpr (requires { algo.sketch_ranks(); }) {
      const auto rank = algo.sketch_ranks();
      const auto mean_rank = algo.sketch_mean_ranks();
      const auto dropouts = algo.sketch_dropouts();
      std::cout << "Position sketch rank last/mean, dropouts: "
                << rank[0] << "/" << mean_rank[0]
                << ", " << dropouts[0] << '\n';
      std::cout << "Curvature direction rank last/mean, dropouts: "
                << rank[1] << "/" << mean_rank[1]
                << ", " << dropouts[1] << '\n';
    }
    if constexpr (requires { algo.mixture_weights(); }) {
      const auto p = algo.mixture_weights();
      std::cout << "Mixture weights diag/position/curvature: "
                << p[0] << " " << p[1] << " " << p[2] << '\n';
    }
    if constexpr (requires { algo.overrelaxation_K(); }) {
      std::cout << "Overrelaxation K: "
                << algo.overrelaxation_K() << '\n';
    }

    HighFive::File h5(output, HighFive::File::Truncate);

    h5.createGroup(model_name);
    h5.createDataSet(std::format("{}/seed", model_name),
                     std::to_string(algo.seed()));
    h5.createDataSet(std::format("{}/draws", model_name), draws);
    h5.createDataSet(std::format("{}/acceptance_rate", model_name), acceptance_rate);
    h5.createDataSet(std::format("{}/log_density", model_name), log_density);
    h5.createDataSet(std::format("{}/nfev", model_name), nfev);
    if constexpr (requires { algo.overrelaxation_K(); }) {
      h5.createDataSet(std::format("{}/K", model_name),
                       static_cast<std::uint64_t>(
                         algo.overrelaxation_K()));
    }

    if constexpr (requires {
      algo.accept_stat();
      algo.divergent();
      algo.n_leapfrog();
      algo.tree_depth();
      algo.energy();
      algo.stepsize();
      algo.variance();
    }) {
      h5.createDataSet(
        std::format("{}/accept_stat", model_name), accept_stat);
      h5.createDataSet(
        std::format("{}/divergent", model_name), divergent);
      h5.createDataSet(
        std::format("{}/n_leapfrog", model_name), n_leapfrog);
      h5.createDataSet(
        std::format("{}/tree_depth", model_name), tree_depth);
      h5.createDataSet(
        std::format("{}/energy", model_name), energy);
      h5.createDataSet(
        std::format("{}/stepsize", model_name), stepsize);
      h5.createDataSet(
        std::format("{}/variance", model_name), algo.variance());
    }

    return 0;
  };

  if (sampler == "normal") {
    klhr::NormalKLHR algo(model, data, klhr_options);
    return run_sampler(algo);
  }

  if (sampler == "slice") {
    klhr::SliceOptions slice_options{
      .seed = seed,
      .warmup = num_warmup,
      .J = J,
      .direction_lowrank_weight = direction_lowrank_weight,
      .direction_min_diag_fraction = direction_min_diag_fraction,
      .lowrank_during_warmup = lowrank_during_warmup,
      .pca_freeze_fraction = pca_freeze_fraction,
    };
    klhr::Slice algo(model, data, slice_options);
    return run_sampler(algo);
  }

  if (sampler == "barker") {
    klhr::BarkerOptions barker_options{
      .seed = seed,
      .warmup = num_warmup,
    };
    klhr::Barker algo(model, data, barker_options);
    return run_sampler(algo);
  }

  if (sampler == "mala") {
    klhr::MALAOptions mala_options{
      .seed = seed,
      .warmup = num_warmup,
    };
    klhr::MALA algo(model, data, mala_options);
    return run_sampler(algo);
  }

  if (sampler == "stan") {
    klhr::StanOptions stan_options{
      .seed = seed,
      .warmup = num_warmup,
      .target_accept = target_accept,
    };
    klhr::Stan algo(model, data, stan_options);
    return run_sampler(algo);
  }

  klhr::SASKLHR algo(model, data, klhr_options);
  return run_sampler(algo);
}
