#include "barker.hpp"
#include "mala.hpp"
#include "normal_klhr.hpp"
#include "sas_klhr.hpp"
#include "slice.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

json parse_json(const std::filesystem::path& config_path, std::size_t index) {
    std::ifstream input(config_path);

    if (!input.is_open()) {
        throw std::runtime_error(
            "Could not open JSON configuration file: " +
            config_path.string()
        );
    }

    if (input.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error(
            "JSON configuration file is empty: " +
            config_path.string()
        );
    }

    json document;

    try {
        input >> document;
    }
    catch (const json::parse_error& error) {
        throw std::runtime_error(
            "Could not parse JSON configuration file '" +
            config_path.string() +
            "': " +
            error.what()
        );
    }

    if (input.bad()) {
        throw std::runtime_error(
            "I/O error while reading JSON configuration file: " +
            config_path.string()
        );
    }

    if (!document.is_array()) {
        throw std::runtime_error(
            "Expected the top-level JSON value in '" +
            config_path.string() +
            "' to be an array"
        );
    }

    if (index >= document.size()) {
        throw std::out_of_range(
            "Configuration index " +
            std::to_string(index) +
            " is outside the valid range 0.." +
            (
                document.empty()
                    ? std::string{"0 (configuration array is empty)"}
                    : std::to_string(document.size() - 1)
            )
        );
    }

    return document.at(index);
}

int main(int argc, char** argv) {

  // defaults
  std::string config = "experiments/config.json";
  std::size_t index = 0;
  std::uint64_t seed = 0;
  std::size_t num_replications = 1;
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
    CLI::App app{"Run a KLHR experiment."};

    app.add_option("--replication", num_replications,
                   "Number of replications (defualt => 1)")
      ->default_val(1)
      ->check(CLI::PositiveNumber);

    app.add_option("--config", config,
                   "Path to config.json")
      ->default_val("experiments/config.json");

    app.add_option("--index", index,
                   "Index number of experiments (default => 0)")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

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
                   "Sampling algorithm: sas, normal, slice, barker, or mala")
      ->check(CLI::IsMember({"sas", "normal", "slice", "barker", "mala"}));

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
  klhr::KlhrOptions klhr_options = {
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

    // std::ifstream f(config);
    // auto cfg = json::parse(f)[index];
    auto cfg = parse_json(config, index);

    std::filesystem::path directory = "output";
    std::filesystem::create_directories(directory);

    std::cout << cfg["iterations"] << std::endl;

    std::string db_path = std::format("output/{}.h5", model_name);
    HighFive::File h5(db_path, HighFive::File::Truncate);

    for (std::size_t r = 0; r < num_replications; ++r) {
      Eigen::Index D = algo.dim();
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
      }

      h5.createGroup(model_name);

      std::string seed_tbl = std::format("{}_{}/seed", model_name, r);
      h5.createDataSet(seed_tbl, std::to_string(algo.seed()));

      std::string draws_tbl = std::format("{}_{}/draws", model_name, r);
      h5.createDataSet(draws_tbl, draws);

      std::string acc_tbl = std::format("{}_{}/acceptance_rate", model_name, r);
      h5.createDataSet(acc_tbl, acceptance_rate);

      std::string ld_tbl = std::format("{}_{}/log_density", model_name, r);
      h5.createDataSet(ld_tbl, log_density);

      std::string nfev_tbl = std::format("{}_{}/nfev", model_name, r);
      h5.createDataSet(nfev_tbl, nfev);
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
      .J = pca_basis,
      .direction_noise_rank = direction_noise_rank,
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

  klhr::SASKLHR algo(model, data, klhr_options);
  return run_sampler(algo);
}
