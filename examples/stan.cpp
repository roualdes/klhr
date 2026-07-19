#include "stan.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char** argv) {

  std::uint64_t seed = 0;
  std::size_t num_warmup = 1'000;
  std::size_t num_iterations = 2'000;
  std::string model_name = "earnings";

  {
    CLI::App app{"Run the standalone Stan-style NUTS sampler."};

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

    CLI11_PARSE(app, argc, argv);
  }

  const std::string model =
    std::format("./stan/{}_model.so", model_name);
  const std::string data = std::format("./stan/{}.json", model_name);
  const klhr::StanOptions options{
    .seed = seed,
    .warmup = num_warmup,
  };

  klhr::Stan sampler(model, data, options);

  Eigen::VectorXd draw;
  for (std::size_t n = 0; n < num_iterations; ++n) {
    draw = sampler.draw();
  }

  std::cout << "Seed: " << sampler.seed() << '\n';
  std::cout << "Final draw: " << draw.transpose() << '\n';
  std::cout << "Stepsize: " << sampler.stepsize() << '\n';
  std::cout << "Diagonal variance: "
            << sampler.variance().transpose() << '\n';
  std::cout << "Number log_density evals: " << sampler.nfev_ << '\n';
  std::cout << "Acceptance statistic: " << sampler.accept_stat() << '\n';
  std::cout << "Mean acceptance statistic: "
            << sampler.acceptance_rate_ << '\n';
  std::cout << "Divergent: " << std::boolalpha
            << sampler.divergent() << '\n';
  std::cout << "Leapfrog steps: " << sampler.n_leapfrog() << '\n';
  std::cout << "Tree depth: " << sampler.tree_depth() << '\n';
  std::cout << "Energy: " << sampler.energy() << '\n';

  return 0;
}
