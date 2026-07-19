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
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

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

constexpr std::uint64_t mix_seed(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::uint64_t make_replication_seed(std::uint64_t base_seed,
                                    std::size_t job_index,
                                    std::size_t replication) {
    const std::uint64_t stream =
        (static_cast<std::uint64_t>(job_index) << 32) |
        static_cast<std::uint64_t>(replication);
    const std::uint64_t seed = mix_seed(base_seed ^ mix_seed(stream));

    // Every sampler reserves zero to request a nondeterministic seed.
    return seed == 0 ? std::numeric_limits<std::uint64_t>::max() : seed;
}

std::filesystem::path make_output_path(const std::filesystem::path& output_root,
                                       const std::string& model_name,
                                       const std::string& sampler_name,
                                       std::size_t index) {
    const std::filesystem::path output_dir =
        output_root / std::to_string(index);

    std::error_code error;
    std::filesystem::create_directories(output_dir, error);

    if (error) {
        throw std::runtime_error(
            "Could not create output directory '" +
            output_dir.string() +
            "': " +
            error.message()
        );
    }

    const auto output_path =
        output_dir / (model_name + "_" + sampler_name + ".h5");

    return output_path;
}

int main(int argc, char** argv) {

  std::string config = "experiments/config.json";
  std::string output_dir = "/app/klhr/output";
  std::size_t index = 0;

  {
    CLI::App app{"Run a KLHR experiment."};

    app.add_option("--config", config,
                   "Path to config.json")
      ->default_val("experiments/config.json");

    app.add_option("--index", index,
                   "Index number of experiments (default => 0)")
      ->default_val(0)
      ->check(CLI::NonNegativeNumber);

    app.add_option("--output-dir", output_dir,
                   "Root directory for experiment output")
      ->default_val("/app/klhr/output");

    CLI11_PARSE(app, argc, argv);
  }

  auto cfg = parse_json(config, index);
  const std::string model_name = cfg.at("model_name").get<std::string>();
  const std::string sampler = cfg.at("sampler").get<std::string>();
  const std::size_t replications = cfg.at("replications").get<std::size_t>();
  const std::uint64_t base_seed = cfg.at("seed").get<std::uint64_t>();
  const std::size_t iterations = cfg.at("iterations").get<std::size_t>();
  const std::size_t warmup = cfg.at("warmup").get<std::size_t>();
  Eigen::Index J = 1;
  if (cfg.contains("J")) {
    J = cfg.at("J").get<Eigen::Index>();
  }
  double target_accept = 0.8;
  if (cfg.contains("target_accept")) {
    target_accept = cfg.at("target_accept").get<double>();
  }

  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions klhr_options = {
    .warmup = warmup,
    .J = J,
  };

  auto run_sampler = [&](auto make_sampler) {

    const auto db_path =
      make_output_path(output_dir, model_name, sampler, index);
    HighFive::File h5(db_path.string(), HighFive::File::Truncate);

    for (std::size_t r = 0; r < replications; ++r) {
      const std::uint64_t replication_seed =
        make_replication_seed(base_seed, index, r);
      auto algo = make_sampler(replication_seed);

      Eigen::Index D = algo.dim();
      Eigen::MatrixXd draws(iterations, D);
      Eigen::VectorXd acceptance_rate(iterations);
      Eigen::VectorXd log_density(iterations);
      Eigen::VectorXd nfev(iterations);
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
        accept_stat.resize(iterations);
        divergent.resize(iterations);
        n_leapfrog.resize(iterations);
        tree_depth.resize(iterations);
        energy.resize(iterations);
        stepsize.resize(iterations);
      }

      Eigen::VectorXd draw(D);

      for (std::size_t n = 0; n < iterations; ++n) {
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
      }

      auto model_tbl = h5.exist(model_name)
        ? h5.getGroup(model_name)
        : h5.createGroup(model_name);

      if (!model_tbl.hasAttribute("iterations")) {
        model_tbl.createAttribute("iterations", iterations);
      }
      if (!model_tbl.hasAttribute("warmup")) {
        model_tbl.createAttribute("warmup", warmup);
      }
      if (!model_tbl.hasAttribute("J")) {
        model_tbl.createAttribute("J", J);
      }

      std::string seed_tbl = std::format("{}/seed/{}", sampler, r);
      model_tbl.createDataSet(seed_tbl, std::to_string(algo.seed()));

      std::string draws_tbl = std::format("{}/draws/{}", sampler, r);
      model_tbl.createDataSet(draws_tbl, draws);

      std::string acc_tbl =
        std::format("{}/acceptance_rate/{}", sampler, r);
      model_tbl.createDataSet(acc_tbl, acceptance_rate);

      std::string ld_tbl =
        std::format("{}/log_density/{}", sampler, r);
      model_tbl.createDataSet(ld_tbl, log_density);

      std::string nfev_tbl = std::format("{}/nfev/{}", sampler, r);
      model_tbl.createDataSet(nfev_tbl, nfev);

      if constexpr (requires {
        algo.accept_stat();
        algo.divergent();
        algo.n_leapfrog();
        algo.tree_depth();
        algo.energy();
        algo.stepsize();
        algo.variance();
      }) {
        model_tbl.createDataSet(
          std::format("{}/accept_stat/{}", sampler, r), accept_stat);
        model_tbl.createDataSet(
          std::format("{}/divergent/{}", sampler, r), divergent);
        model_tbl.createDataSet(
          std::format("{}/n_leapfrog/{}", sampler, r), n_leapfrog);
        model_tbl.createDataSet(
          std::format("{}/tree_depth/{}", sampler, r), tree_depth);
        model_tbl.createDataSet(
          std::format("{}/energy/{}", sampler, r), energy);
        model_tbl.createDataSet(
          std::format("{}/stepsize/{}", sampler, r), stepsize);
        model_tbl.createDataSet(
          std::format("{}/variance/{}", sampler, r), algo.variance());
      }
    }

    return 0;
  };

  if (sampler == "normal") {
    return run_sampler([&](const std::uint64_t replication_seed) {
      auto options = klhr_options;
      options.seed = replication_seed;
      return klhr::NormalKLHR(model, data, options);
    });
  }

  if (sampler == "slice") {
    klhr::SliceOptions slice_options{
      .warmup = warmup,
      .J = J,
    };
    return run_sampler([&](const std::uint64_t replication_seed) {
      auto options = slice_options;
      options.seed = replication_seed;
      return klhr::Slice(model, data, options);
    });
  }

  if (sampler == "barker") {
    klhr::BarkerOptions barker_options{
      .warmup = warmup,
    };
    return run_sampler([&](const std::uint64_t replication_seed) {
      auto options = barker_options;
      options.seed = replication_seed;
      return klhr::Barker(model, data, options);
    });
  }

  if (sampler == "mala") {
    klhr::MALAOptions mala_options{
      .warmup = warmup,
    };
    return run_sampler([&](const std::uint64_t replication_seed) {
      auto options = mala_options;
      options.seed = replication_seed;
      return klhr::MALA(model, data, options);
    });
  }

  if (sampler == "stan") {
    klhr::StanOptions stan_options{
      .warmup = warmup,
      .target_accept = target_accept,
    };
    return run_sampler([&](const std::uint64_t replication_seed) {
      auto options = stan_options;
      options.seed = replication_seed;
      return klhr::Stan(model, data, options);
    });
  }

  return run_sampler([&](const std::uint64_t replication_seed) {
    auto options = klhr_options;
    options.seed = replication_seed;
    return klhr::SASKLHR(model, data, options);
  });
}
