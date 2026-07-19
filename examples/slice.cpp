#include "slice.hpp"

#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include <welford.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::uint64_t seed = 0;
  std::size_t warmup = 15'000;
  std::size_t iterations = 30'000;
  std::string model_name = "earnings";
  std::string model_library;
  std::string data_file;
  double initial_width = 1.0;
  std::size_t max_steps_out = 100;
  std::size_t max_shrink_steps = 1'000;

  CLI::App app{"Run random-direction slice sampling."};
  app.add_option("--seed", seed, "Random seed (0 => random)")
    ->default_val(seed)
    ->check(CLI::NonNegativeNumber);
  app.add_option("--warmup", warmup, "Number of warmup iterations")
    ->default_val(warmup)
    ->check(CLI::NonNegativeNumber);
  app.add_option("--iterations", iterations,
                 "Number of total iterations (warmup included)")
    ->default_val(iterations)
    ->check(CLI::NonNegativeNumber);
  app.add_option("--model", model_name, "Stan model name")
    ->default_val(model_name);
  app.add_option("--model-library", model_library,
                 "Explicit Stan model library path");
  app.add_option("--data", data_file, "Explicit Stan data JSON path");
  app.add_option("--initial-width", initial_width,
                 "Slice bracket width multiplier")
    ->default_val(initial_width);
  app.add_option("--max-steps-out", max_steps_out,
                 "Maximum stepping-out intervals")
    ->default_val(max_steps_out)
    ->check(CLI::PositiveNumber);
  app.add_option("--max-shrink-steps", max_shrink_steps,
                 "Maximum shrinkage proposals")
    ->default_val(max_shrink_steps)
    ->check(CLI::PositiveNumber);
  CLI11_PARSE(app, argc, argv);

  const std::string model = model_library.empty() ?
    std::format("./stan/{}_model.so", model_name) : model_library;
  const std::string data = data_file.empty() ?
    std::format("./stan/{}.json", model_name) : data_file;
  klhr::SliceOptions options{
    .seed = seed,
    .warmup = warmup,
    .initial_width = initial_width,
    .max_steps_out = max_steps_out,
    .max_shrink_steps = max_shrink_steps,
  };
  klhr::Slice sampler(model, data, options);

  Eigen::VectorXd draw;
  mcmcpp::WelfordAccumulator moments(sampler.dim());
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    draw = sampler.draw();
    if (iteration >= warmup) {
      moments.update(draw);
    }
  }

  const Eigen::VectorXd mean = moments.mean();
  const Eigen::VectorXd variance = moments.variance();
  std::cout << "Seed: " << sampler.seed() << '\n';
  std::cout << "Final draw: " << draw.transpose() << '\n';
  std::cout << "RMS component mean: "
            << mean.norm() / std::sqrt(sampler.dim()) << '\n';
  std::cout << "Mean component variance: " << variance.mean() << '\n';
  std::cout << "Last width: " << sampler.width() << '\n';
  std::cout << "Number log_density evals: " << sampler.nfev_ << '\n';
  std::cout << "Successful slice updates: "
            << sampler.acceptance_rate_ << '\n';
}
