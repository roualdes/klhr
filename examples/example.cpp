#include "base_klhr.hpp"
#include "normal_klhr.hpp"
#include "sas_klhr.hpp"
#include "welford.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

int main(int argc, char** argv) {

  // defaults
  std::uint64_t seed = 0;
  std::size_t num_warmup = 15'000;
  std::size_t num_iterations = 30'000;
  std::string model_name = "earnings";
  std::string sampler = "sas";
  Eigen::Index pca_basis = klhr::KlhrOptions{}.J;
  Eigen::Index direction_noise_rank = klhr::KlhrOptions{}.direction_noise_rank;
  double direction_lowrank_weight = klhr::KlhrOptions{}.direction_lowrank_weight;
  double direction_min_diag_fraction =
    klhr::KlhrOptions{}.direction_min_diag_fraction;
  bool lowrank_during_warmup = klhr::KlhrOptions{}.lowrank_during_warmup;
  double pca_freeze_fraction = klhr::KlhrOptions{}.pca_freeze_fraction;
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

  {
    CLI::App app{"Run KLHR."};

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
                   "Path to the Stan model library (.so from BridgeStan)");

    app.add_option("--sampler", sampler,
                   "Name of sampling algorithm to use");

    app.add_option("--pca-basis", pca_basis,
                   "Number of PCA basis vectors to learn")
      ->default_val(pca_basis);

    app.add_option("--direction-noise-rank", direction_noise_rank,
                   "Rank of learned PCA covariance used in regular direction noise (negative => J)")
      ->default_val(direction_noise_rank);

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

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions options = {
    .seed = seed,
    .warmup = num_warmup,
    .J = pca_basis,
    .direction_noise_rank = direction_noise_rank,
    .direction_lowrank_weight = direction_lowrank_weight,
    .direction_min_diag_fraction = direction_min_diag_fraction,
    .lowrank_during_warmup = lowrank_during_warmup,
    .pca_freeze_fraction = pca_freeze_fraction,
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
  };

  auto run_sampler = [&](auto& algo) {
    using Algo = std::remove_cvref_t<decltype(algo)>;

    Eigen::Index D = algo.dim();
    mcmcpp::WelfordAccumulator w{D};

    Eigen::MatrixXd draws(num_iterations, D);
    Eigen::VectorXd acceptance_rate(num_iterations);
    Eigen::VectorXd log_density(num_iterations);
    Eigen::VectorXd nfev(num_iterations);

    Eigen::VectorXd draw(D);
    for (std::size_t n = 0; n < num_iterations; ++n) {
      draw = algo.draw();
      draws.row(n) = draw;
      acceptance_rate(n) = algo.acceptance_rate_;
      log_density(n) = algo.log_density_;
      nfev(n) = algo.nfev_;
      if (n >= num_warmup) {
        w.update(draw);
      }
    }

    HighFive::File h5("draws/experiments.h5", HighFive::File::Truncate);

    h5.createGroup(model_name);
    h5.createDataSet(std::format("{}/seed", model_name),
                     std::to_string(algo.seed()));
    h5.createDataSet(std::format("{}/draws", model_name), draws);
    h5.createDataSet(std::format("{}/acceptance_rate", model_name), acceptance_rate);
    h5.createDataSet(std::format("{}/log_density", model_name), log_density);
    h5.createDataSet(std::format("{}/nfev", model_name), nfev);

    auto vector_to_eigen = [](const std::vector<double>& x) {
      Eigen::VectorXd out(x.size());
      for (Eigen::Index i = 0; i < out.size(); ++i) {
        out(i) = x[static_cast<std::size_t>(i)];
      }
      return out;
    };

    auto vector_to_matrix = [D](const std::vector<Eigen::VectorXd>& x) {
      Eigen::MatrixXd out(x.size(), D);
      for (Eigen::Index i = 0; i < out.rows(); ++i) {
        out.row(i) = x[static_cast<std::size_t>(i)].transpose();
      }
      return out;
    };

    h5.createDataSet(std::format("{}/proposal_draws", model_name),
                     vector_to_matrix(algo.proposal_draw_history()));
    h5.createDataSet(std::format("{}/proposal_log_accept", model_name),
                     vector_to_eigen(algo.proposal_log_accept_history()));
    h5.createDataSet(std::format("{}/proposal_log_density", model_name),
                     vector_to_eigen(algo.proposal_log_density_history()));
    h5.createDataSet(std::format("{}/proposal_accepted", model_name),
                     vector_to_eigen(algo.proposal_accepted_history()));
    h5.createDataSet(std::format("{}/proposal_valid", model_name),
                     vector_to_eigen(algo.proposal_valid_history()));

    if constexpr (std::is_base_of_v<klhr::BaseKLHR, Algo>) {
      h5.createDataSet(std::format("{}/transport_distance", model_name),
                       vector_to_eigen(algo.transport_distance_history()));
      h5.createDataSet(std::format("{}/transport_reflections", model_name),
                       vector_to_eigen(algo.transport_reflections_history()));
      h5.createDataSet(std::format("{}/transport_logp_gain", model_name),
                       vector_to_eigen(algo.transport_logp_gain_history()));
      h5.createDataSet(std::format("{}/transport_uturn", model_name),
                       vector_to_eigen(algo.transport_uturn_history()));
      h5.createDataSet(std::format("{}/transport_moved", model_name),
                       vector_to_eigen(algo.transport_moved_history()));
      h5.createDataSet(std::format("{}/transport_direction_norm", model_name),
                       vector_to_eigen(algo.transport_direction_norm_history()));
      h5.createDataSet(std::format("{}/transport_variance", model_name),
                       vector_to_matrix(algo.transport_variance_history()));
      h5.createDataSet(std::format("{}/transport_handoff_pca_basis", model_name),
                       algo.transport_handoff_pca_basis());
      h5.createDataSet(std::format("{}/transport_handoff_pca_weights", model_name),
                       algo.transport_handoff_pca_weights());
      h5.createDataSet(std::format("{}/transport_handoff_pca_count", model_name),
                       static_cast<double>(algo.transport_handoff_pca_count()));
      h5.createDataSet(std::format("{}/transport_handoff_pca_ready", model_name),
                       algo.transport_handoff_pca_ready() ? 1.0 : 0.0);
      h5.createDataSet(std::format("{}/transport_handoff_pca_whitened", model_name),
                       algo.transport_handoff_pca_whitened() ? 1.0 : 0.0);
      h5.createDataSet(std::format("{}/transport_rollback", model_name),
                       algo.transport_rollback() ? 1.0 : 0.0);
      h5.createDataSet(std::format("{}/transport_initial_log_density", model_name),
                       algo.transport_initial_log_density());
      h5.createDataSet(std::format("{}/transport_best_log_density", model_name),
                       algo.transport_best_log_density());
      h5.createDataSet(std::format("{}/transport_endpoint_from_best_drop", model_name),
                       algo.transport_endpoint_from_best_drop());
    }

    if constexpr (std::is_same_v<Algo, klhr::SASKLHR>) {
      h5.createDataSet(std::format("{}/sas_m", model_name),
                       vector_to_eigen(algo.sas_m_history()));
      h5.createDataSet(std::format("{}/sas_sampled_xi", model_name),
                       vector_to_eigen(algo.sas_xi_history()));
      h5.createDataSet(std::format("{}/sas_accepted_xi", model_name),
                       vector_to_eigen(algo.sas_accepted_xi_history()));
      h5.createDataSet(std::format("{}/sas_accepted", model_name),
                       vector_to_eigen(algo.sas_accepted_history()));
    }

    mcmcpp::WelfordAccumulator msjd{};
    for (std::size_t n = 0; n < num_iterations - 1; ++n) {
      if (n > num_warmup) {
        msjd.update((draws.row(n + 1) - draws.row(n)).norm());
      }
    }

    std::cout << "Seed: " << algo.seed() << '\n';
    std::cout << "means: " << w.mean().transpose() << '\n';
    std::cout << "stds: " << w.std().transpose() << '\n';
    std::cout << "msjd: " << msjd.mean()(0) << '\n';
    std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
    std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';

    return 0;
  };

  if (sampler == "normal") {
    klhr::NormalKLHR algo(model, data, options);
    return run_sampler(algo);
  }

  klhr::SASKLHR algo(model, data, options);
  return run_sampler(algo);
}
