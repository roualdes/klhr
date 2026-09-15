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
  std::string model_name = "normal";
  std::string sampler = "sas";
  std::string output = "draws/experiments.h5";
  double target_accept = 0.8;
  double laplace_kl_residual_tol =
    klhr::KlhrOptions{}.laplace_kl_residual_tol;
  std::size_t K = klhr::KlhrOptions{}.K;
  // Slice arguments, at Neal (2003) defaults.
  double slice_width = klhr::SliceOptions{}.initial_width;
  std::size_t max_steps_out = klhr::SliceOptions{}.max_steps_out;
  std::size_t max_shrink_steps = klhr::SliceOptions{}.max_shrink_steps;
  double min_width = klhr::SliceOptions{}.min_width;
  double max_width = klhr::SliceOptions{}.max_width;

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

    app.add_option("--target_accept", target_accept)
      ->default_val(0.8)
      ->check(CLI::PositiveNumber);

    app.add_option("--laplace-kl-residual-tol", laplace_kl_residual_tol,
                   "Accept the Laplace line fit when its KL gradient residual is below this")
      ->default_val(laplace_kl_residual_tol)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--K", K,
                   "Ordered overrelaxation level (rounded down to odd)")
      ->default_val(K)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--slice-width", slice_width,
                   "Slice step size w, scaled by the metric (Neal: 1)")
      ->default_val(slice_width)->check(CLI::PositiveNumber);

    app.add_option("--max-steps-out", max_steps_out,
                   "Slice stepping-out limit m, 0 = unlimited (Neal: 0)")
      ->default_val(max_steps_out)->check(CLI::NonNegativeNumber);

    app.add_option("--max-shrink-steps", max_shrink_steps,
                   "Slice shrinkage limit, 0 = unlimited (Neal: 0)")
      ->default_val(max_shrink_steps)->check(CLI::NonNegativeNumber);

    app.add_option("--min-width", min_width,
                   "Floor on the metric-scaled slice width")
      ->default_val(min_width)->check(CLI::PositiveNumber);

    app.add_option("--max-width", max_width,
                   "Cap on the metric-scaled slice width")
      ->default_val(max_width)->check(CLI::PositiveNumber);

    CLI11_PARSE(app, argc, argv);
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions klhr_options = {
    .seed = seed,
    .laplace_kl_residual_tol = laplace_kl_residual_tol,
    .K = K,
    .warmup = num_warmup,
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
      if (mv.size() <= 1200) {
        std::cout << "Adapted diagonal: " << mv.transpose() << '\n';
      }
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
      .initial_width = slice_width,
      .max_steps_out = max_steps_out,
      .max_shrink_steps = max_shrink_steps,
      .min_width = min_width,
      .max_width = max_width,
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
