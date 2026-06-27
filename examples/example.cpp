#include "base_klhr.hpp"
#include "normal_klhr.hpp"
#include "sas_klhr.hpp"
#include "welford.hpp"

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

namespace {

enum class SamplerKind {
  Sas,
  Normal
};

struct RunConfig {
  std::size_t N = 30'000;
  std::size_t warmup = 15'000;
  std::uint64_t seed = 0;
  std::size_t transport_steps = 100;
  std::size_t gradient_history = 3;
  double projection_probability = 0.5;
  double transport_kappa = 5.0;
  std::string output = "draws/earnings.h5";
  SamplerKind sampler = SamplerKind::Sas;
  klhr::TransportDirectionLaw transport_direction_law =
    klhr::TransportDirectionLaw::Kappa;
  klhr::TransportProposal transport_proposal =
    klhr::TransportProposal::Random;
};

std::string require_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::format("missing value for {}", argv[i]));
  }
  ++i;
  return argv[i];
}

SamplerKind parse_sampler_kind(const std::string& value) {
  if (value == "sas") {
    return SamplerKind::Sas;
  }
  if (value == "normal") {
    return SamplerKind::Normal;
  }
  throw std::runtime_error("sampler must be sas or normal");
}

klhr::TransportProposal parse_transport_proposal(const std::string& value) {
  if (value == "overrelaxed") {
    return klhr::TransportProposal::Overrelaxed;
  }
  if (value == "random") {
    return klhr::TransportProposal::Random;
  }
  throw std::runtime_error("transport proposal must be overrelaxed or random");
}

klhr::TransportDirectionLaw parse_transport_direction_law(
    const std::string& value) {
  if (value == "kappa") {
    return klhr::TransportDirectionLaw::Kappa;
  }
  if (value == "projected") {
    return klhr::TransportDirectionLaw::Projected;
  }
  throw std::runtime_error("transport direction must be kappa or projected");
}

RunConfig parse_args(int argc, char** argv) {
  RunConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output") {
      config.output = require_value(i, argc, argv);
    } else if (arg == "--sampler") {
      config.sampler = parse_sampler_kind(require_value(i, argc, argv));
    } else if (arg == "--transport-approx") {
      config.sampler = parse_sampler_kind(require_value(i, argc, argv));
    } else if (arg == "--transport-proposal") {
      config.transport_proposal =
        parse_transport_proposal(require_value(i, argc, argv));
    } else if (arg == "--transport-direction") {
      config.transport_direction_law =
        parse_transport_direction_law(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      config.seed = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--n") {
      config.N = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--warmup") {
      config.warmup = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--transport-steps") {
      config.transport_steps = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--gradient-history") {
      config.gradient_history = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--projection-probability") {
      config.projection_probability = std::stod(require_value(i, argc, argv));
    } else if (arg == "--transport-kappa") {
      config.transport_kappa = std::stod(require_value(i, argc, argv));
    } else {
      throw std::runtime_error(std::format("unknown argument {}", arg));
    }
  }
  return config;
}

} // namespace

int main(int argc, char** argv) {

  const RunConfig config = parse_args(argc, argv);
  std::string model_name = "earnings";
  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KlhrOptions options;
  options.warmup = config.warmup;
  options.seed = config.seed;
  options.initial_fast_adaptation_steps = config.transport_steps;
  options.initial_transport_gradient_history = config.gradient_history;
  options.initial_transport_gradient_projection_probability =
    config.projection_probability;
  options.initial_transport_direction_kappa = config.transport_kappa;
  options.initial_transport_direction_law = config.transport_direction_law;
  options.initial_transport_proposal = config.transport_proposal;
  std::unique_ptr<klhr::BaseKLHR> algo;
  if (config.sampler == SamplerKind::Normal) {
    algo = std::make_unique<klhr::NormalKLHR>(model, data, options);
  } else {
    algo = std::make_unique<klhr::SASKLHR>(model, data, options);
  }
  std::size_t D = algo->dim();
  WelfordAccumulator w{D};

  Eigen::MatrixXd draws(config.N, D);
  Eigen::VectorXd acceptance_rate(config.N);
  Eigen::VectorXd log_density(config.N);
  Eigen::VectorXd nfev(config.N);
  Eigen::VectorXd diagnostic_phase(config.N);
  Eigen::VectorXd grad_dot_move(config.N);
  Eigen::VectorXd cos_grad_move(config.N);
  Eigen::VectorXd beta_slope(config.N);
  Eigen::VectorXd diagnostic_logp_gain(config.N);
  Eigen::VectorXd diag_jump(config.N);
  Eigen::VectorXd move_norm(config.N);
  Eigen::VectorXd grad_norm(config.N);
  Eigen::VectorXd transport_direction_attempts(config.N);
  Eigen::MatrixXd diagnostic_gradient(config.N, D);
  Eigen::MatrixXd diagnostic_move(config.N, D);

  const std::filesystem::path output_path{config.output};
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  HighFive::File h5(config.output, HighFive::File::Truncate);

  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < config.N; ++n) {
    draw = algo->draw();
    draws.row(n) = draw;
    acceptance_rate(n) = algo->acceptance_rate_;
    log_density(n) = algo->log_density_;
    nfev(n) = algo->nfev_;
    diagnostic_phase(n) = static_cast<double>(algo->diagnostic_phase_);
    grad_dot_move(n) = algo->diagnostic_grad_dot_move_;
    cos_grad_move(n) = algo->diagnostic_cos_grad_move_;
    beta_slope(n) = algo->diagnostic_beta_slope_;
    diagnostic_logp_gain(n) = algo->diagnostic_logp_gain_;
    diag_jump(n) = algo->diagnostic_diag_jump_;
    move_norm(n) = algo->diagnostic_move_norm_;
    grad_norm(n) = algo->diagnostic_grad_norm_;
    transport_direction_attempts(n) =
      static_cast<double>(algo->diagnostic_transport_direction_attempts_);
    diagnostic_gradient.row(n) = algo->diagnostic_gradient_.transpose();
    diagnostic_move.row(n) = algo->diagnostic_move_.transpose();
    if (n >= config.warmup) {
      w.update(draw);
    }
  }

  h5.createGroup("earnings");
  h5.createGroup("earnings/diagnostics");
  h5.createDataSet("earnings/draws", draws);
  h5.createDataSet("earnings/acceptance_rate", acceptance_rate);
  h5.createDataSet("earnings/log_density", log_density);
  h5.createDataSet("earnings/nfev", nfev);
  h5.createDataSet("earnings/stop_transport_idx", algo->stop_transport_idx_);
  h5.createDataSet("earnings/diagnostics/phase", diagnostic_phase);
  h5.createDataSet("earnings/diagnostics/grad_dot_move", grad_dot_move);
  h5.createDataSet("earnings/diagnostics/cos_grad_move", cos_grad_move);
  h5.createDataSet("earnings/diagnostics/beta_slope", beta_slope);
  h5.createDataSet("earnings/diagnostics/logp_gain", diagnostic_logp_gain);
  h5.createDataSet("earnings/diagnostics/diag_jump", diag_jump);
  h5.createDataSet("earnings/diagnostics/move_norm", move_norm);
  h5.createDataSet("earnings/diagnostics/grad_norm", grad_norm);
  h5.createDataSet("earnings/diagnostics/transport_direction_attempts",
                   transport_direction_attempts);
  h5.createDataSet("earnings/diagnostics/gradient_unconstrained",
                   diagnostic_gradient);
  h5.createDataSet("earnings/diagnostics/move_unconstrained",
                   diagnostic_move);

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "Number log_density evals: " << algo->nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo->acceptance_rate_ << '\n';
  std::cout << "Output: " << config.output << '\n';

  return 0;
}
