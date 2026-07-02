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
#include <format>
#include <iostream>
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
  std::size_t initial_transport_steps = 750;
  std::size_t transport_max_leapfrog_steps = 500;
  std::size_t transport_stepsize_search_steps = 20;
  double transport_target_accept = 0.8;
  double transport_initial_stepsize = 1.0;
  double transport_min_stepsize = 1e-8;
  double transport_max_stepsize = 1e6;
  double transport_max_delta_energy = 1000.0;
  double transport_momentum_persistence = 0.9;
  double transport_failure_momentum_decay = 0.5;
  double transport_max_momentum_norm = 10.0;

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

    app.add_option("--initial-transport-steps", initial_transport_steps,
                   "Initial nonstationary HMC-style transport iterations")
      ->default_val(initial_transport_steps)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--transport-max-leapfrog-steps", transport_max_leapfrog_steps,
                   "Maximum leapfrog steps per initial transport iteration")
      ->default_val(transport_max_leapfrog_steps)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--transport-stepsize-search-steps",
                   transport_stepsize_search_steps,
                   "Maximum step-size search attempts per initial transport iteration")
      ->default_val(transport_stepsize_search_steps)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--transport-target-accept", transport_target_accept,
                   "One-step Hamiltonian acceptance target for initial transport")
      ->default_val(transport_target_accept);

    app.add_option("--transport-initial-stepsize", transport_initial_stepsize,
                   "Initial leapfrog step size for initial transport")
      ->default_val(transport_initial_stepsize);

    app.add_option("--transport-min-stepsize", transport_min_stepsize,
                   "Minimum leapfrog step size for initial transport")
      ->default_val(transport_min_stepsize);

    app.add_option("--transport-max-stepsize", transport_max_stepsize,
                   "Maximum leapfrog step size for initial transport")
      ->default_val(transport_max_stepsize);

    app.add_option("--transport-max-delta-energy", transport_max_delta_energy,
                   "Divergence guard for initial transport Hamiltonian error")
      ->default_val(transport_max_delta_energy);

    app.add_option("--transport-momentum-persistence",
                   transport_momentum_persistence,
                   "Partial momentum refresh persistence for initial transport")
      ->default_val(transport_momentum_persistence);

    app.add_option("--transport-failure-momentum-decay",
                   transport_failure_momentum_decay,
                   "Momentum flip/damping factor after failed initial transport")
      ->default_val(transport_failure_momentum_decay);

    app.add_option("--transport-max-momentum-norm", transport_max_momentum_norm,
                   "Maximum standardized persistent momentum norm in initial transport")
      ->default_val(transport_max_momentum_norm);

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions options = {
    .seed = seed,
    .warmup = num_warmup,
    .initial_transport_steps = sampler == "sas" ? initial_transport_steps : 0,
    .transport_max_leapfrog_steps = transport_max_leapfrog_steps,
    .transport_stepsize_search_steps = transport_stepsize_search_steps,
    .transport_target_accept = transport_target_accept,
    .transport_initial_stepsize = transport_initial_stepsize,
    .transport_min_stepsize = transport_min_stepsize,
    .transport_max_stepsize = transport_max_stepsize,
    .transport_max_delta_energy = transport_max_delta_energy,
    .transport_momentum_persistence = transport_momentum_persistence,
    .transport_failure_momentum_decay = transport_failure_momentum_decay,
    .transport_max_momentum_norm = transport_max_momentum_norm,
  };

  auto run_sampler = [&](auto& algo) {
    using Algo = std::remove_cvref_t<decltype(algo)>;

    std::size_t D = algo.dim();
    WelfordAccumulator w{D};

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
      h5.createDataSet(std::format("{}/transport_stepsize", model_name),
                       vector_to_eigen(algo.transport_stepsize_history()));
      h5.createDataSet(std::format("{}/transport_leapfrog_steps", model_name),
                       vector_to_eigen(algo.transport_leapfrog_steps_history()));
      h5.createDataSet(std::format("{}/transport_accept_stat", model_name),
                       vector_to_eigen(algo.transport_accept_stat_history()));
      h5.createDataSet(std::format("{}/transport_uturn", model_name),
                       vector_to_eigen(algo.transport_uturn_history()));
      h5.createDataSet(std::format("{}/transport_moved", model_name),
                       vector_to_eigen(algo.transport_moved_history()));
      h5.createDataSet(std::format("{}/transport_momentum_norm", model_name),
                       vector_to_eigen(algo.transport_momentum_norm_history()));
      h5.createDataSet(std::format("{}/transport_variance", model_name),
                       vector_to_matrix(algo.transport_variance_history()));
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

    std::cout << "means: " << w.mean().transpose() << '\n';
    std::cout << "stds: " << w.std().transpose() << '\n';
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
