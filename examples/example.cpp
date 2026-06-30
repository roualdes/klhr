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
#include <utility>

int main(int argc, char** argv) {

  // defaults
  std::uint64_t seed = 0;
  std::size_t num_warmup = 15'000;
  std::size_t num_iterations = 30'000;
  std::string model_name = "earnings";
  std::string sampler = "sas";

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

    app.add_option("model", model_name,
                   "Path to the Stan model library (.so from BridgeStan)");

    app.add_option("sampler", sampler,
                   "Name of sampling algorithm to use");

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions options = {
    .seed = seed,
    .warmup = num_warmup,
  };

  std::unique_ptr<klhr::BaseKLHR> algo;
  if (sampler == "normal") {
    algo = std::make_unique<klhr::NormalKLHR>(model, data, options);
  } else {
    algo = std::make_unique<klhr::SASKLHR>(model, data, options);
  }
  std::size_t D = algo->dim();
  WelfordAccumulator w{D};

  Eigen::MatrixXd draws(num_iterations, D);
  Eigen::VectorXd acceptance_rate(num_iterations);
  Eigen::VectorXd log_density(num_iterations);
  Eigen::VectorXd nfev(num_iterations);

  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < num_iterations; ++n) {
    draw = algo->draw();
    draws.row(n) = draw;
    acceptance_rate(n) = algo->acceptance_rate_;
    log_density(n) = algo->log_density_;
    nfev(n) = algo->nfev_;
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

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "Number log_density evals: " << algo->nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo->acceptance_rate_ << '\n';

  return 0;
}
