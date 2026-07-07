#include "mala.hpp"
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

  {
    CLI::App app{"Run MALA."};

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

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::MALAOptions options = {
    .seed = seed,
    .warmup = num_warmup,
  };

  klhr::MALA algo(model, data, options);

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

  mcmcpp::WelfordAccumulator msjd{};
  for (std::size_t n = 0; n < num_iterations - 1; ++n) {
    if (n > num_warmup) {
      msjd.update((draws.row(n + 1) - draws.row(n)).norm());
    }
  }

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "msjd: " << msjd.mean()(0) << '\n';
  std::cout << "stepsize: " << algo.stepsize() << '\n';
  std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';

  return 0;
}
