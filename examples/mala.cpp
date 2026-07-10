#include "mala.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>

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

  Eigen::VectorXd draw;
  for (std::size_t n = 0; n < num_iterations; ++n) {
    draw = algo.draw();
  }

  std::cout << "Final draw: " << draw.transpose() << '\n';
  std::cout << "stepsize: " << algo.stepsize() << '\n';
  std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';

  return 0;
}
