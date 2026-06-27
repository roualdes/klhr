#include "klhr.hpp"
#include "welford.hpp"

#include <Eigen/Dense>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

#include <cstddef>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct RunConfig {
  std::size_t N = 30'000;
  std::size_t warmup = 15'000;
  std::uint64_t seed = 0;
  std::size_t gradient_history = 3;
  double projection_probability = 0.5;
  double transport_kappa = 5.0;
  std::string output = "draws/earnings.h5";
  klhr::TransportApproximation transport_approximation =
    klhr::TransportApproximation::Sas;
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

klhr::TransportApproximation parse_transport_approximation(
    const std::string& value) {
  if (value == "sas") {
    return klhr::TransportApproximation::Sas;
  }
  if (value == "normal") {
    return klhr::TransportApproximation::Normal;
  }
  throw std::runtime_error("transport approximation must be sas or normal");
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

RunConfig parse_args(int argc, char** argv) {
  RunConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output") {
      config.output = require_value(i, argc, argv);
    } else if (arg == "--transport-approx") {
      config.transport_approximation =
        parse_transport_approximation(require_value(i, argc, argv));
    } else if (arg == "--transport-proposal") {
      config.transport_proposal =
        parse_transport_proposal(require_value(i, argc, argv));
    } else if (arg == "--seed") {
      config.seed = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--n") {
      config.N = std::stoull(require_value(i, argc, argv));
    } else if (arg == "--warmup") {
      config.warmup = std::stoull(require_value(i, argc, argv));
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
  options.initial_transport_gradient_history = config.gradient_history;
  options.initial_transport_gradient_projection_probability =
    config.projection_probability;
  options.initial_transport_direction_kappa = config.transport_kappa;
  options.initial_transport_approximation = config.transport_approximation;
  options.initial_transport_proposal = config.transport_proposal;
  klhr::KLHR algo{model, data, options};
  std::size_t D = algo.dim();
  std::size_t C = algo.diagnostic_candidate_count_;
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
  Eigen::VectorXd jump_bonus(config.N);
  Eigen::VectorXd diag_jump(config.N);
  Eigen::VectorXd move_norm(config.N);
  Eigen::VectorXd grad_norm(config.N);
  Eigen::VectorXd selected_candidate(config.N);
  Eigen::MatrixXd diagnostic_gradient(config.N, D);
  Eigen::MatrixXd diagnostic_move(config.N, D);
  Eigen::MatrixXd candidate_log_weight(config.N, C);
  Eigen::MatrixXd candidate_probability(config.N, C);
  Eigen::MatrixXd candidate_logp_gain(config.N, C);
  Eigen::MatrixXd candidate_jump_bonus(config.N, C);
  Eigen::MatrixXd candidate_diag_jump(config.N, C);
  Eigen::MatrixXd candidate_move_norm(config.N, C);
  Eigen::MatrixXd candidate_grad_dot_move(config.N, C);
  Eigen::MatrixXd candidate_cos_grad_move(config.N, C);
  Eigen::MatrixXd candidate_beta_slope(config.N, C);
  Eigen::MatrixXd candidate_delta_beta0(config.N, C);
  Eigen::MatrixXd candidate_delta_beta1(config.N, C);

  const std::filesystem::path output_path{config.output};
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  HighFive::File h5(config.output, HighFive::File::Truncate);

  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < config.N; ++n) {
    draw = algo.draw();
    draws.row(n) = draw;
    acceptance_rate(n) = algo.acceptance_rate_;
    log_density(n) = algo.log_density_;
    nfev(n) = algo.nfev_;
    diagnostic_phase(n) = static_cast<double>(algo.diagnostic_phase_);
    grad_dot_move(n) = algo.diagnostic_grad_dot_move_;
    cos_grad_move(n) = algo.diagnostic_cos_grad_move_;
    beta_slope(n) = algo.diagnostic_beta_slope_;
    diagnostic_logp_gain(n) = algo.diagnostic_logp_gain_;
    jump_bonus(n) = algo.diagnostic_jump_bonus_;
    diag_jump(n) = algo.diagnostic_diag_jump_;
    move_norm(n) = algo.diagnostic_move_norm_;
    grad_norm(n) = algo.diagnostic_grad_norm_;
    selected_candidate(n) = static_cast<double>(algo.diagnostic_selected_candidate_);
    diagnostic_gradient.row(n) = algo.diagnostic_gradient_.transpose();
    diagnostic_move.row(n) = algo.diagnostic_move_.transpose();
    candidate_log_weight.row(n) =
      algo.diagnostic_candidate_log_weight_.transpose();
    candidate_probability.row(n) =
      algo.diagnostic_candidate_probability_.transpose();
    candidate_logp_gain.row(n) =
      algo.diagnostic_candidate_logp_gain_.transpose();
    candidate_jump_bonus.row(n) =
      algo.diagnostic_candidate_jump_bonus_.transpose();
    candidate_diag_jump.row(n) =
      algo.diagnostic_candidate_diag_jump_.transpose();
    candidate_move_norm.row(n) =
      algo.diagnostic_candidate_move_norm_.transpose();
    candidate_grad_dot_move.row(n) =
      algo.diagnostic_candidate_grad_dot_move_.transpose();
    candidate_cos_grad_move.row(n) =
      algo.diagnostic_candidate_cos_grad_move_.transpose();
    candidate_beta_slope.row(n) =
      algo.diagnostic_candidate_beta_slope_.transpose();
    candidate_delta_beta0.row(n) =
      algo.diagnostic_candidate_delta_beta0_.transpose();
    candidate_delta_beta1.row(n) =
      algo.diagnostic_candidate_delta_beta1_.transpose();
    if (n >= config.warmup) {
      w.update(draw);
    }
  }

  h5.createGroup("earnings");
  h5.createGroup("earnings/diagnostics");
  h5.createGroup("earnings/diagnostics/transport_candidates");
  h5.createDataSet("earnings/draws", draws);
  h5.createDataSet("earnings/acceptance_rate", acceptance_rate);
  h5.createDataSet("earnings/log_density", log_density);
  h5.createDataSet("earnings/nfev", nfev);
  h5.createDataSet("earnings/stop_transport_idx", algo.stop_transport_idx_);
  h5.createDataSet("earnings/diagnostics/phase", diagnostic_phase);
  h5.createDataSet("earnings/diagnostics/grad_dot_move", grad_dot_move);
  h5.createDataSet("earnings/diagnostics/cos_grad_move", cos_grad_move);
  h5.createDataSet("earnings/diagnostics/beta_slope", beta_slope);
  h5.createDataSet("earnings/diagnostics/logp_gain", diagnostic_logp_gain);
  h5.createDataSet("earnings/diagnostics/jump_bonus", jump_bonus);
  h5.createDataSet("earnings/diagnostics/diag_jump", diag_jump);
  h5.createDataSet("earnings/diagnostics/move_norm", move_norm);
  h5.createDataSet("earnings/diagnostics/grad_norm", grad_norm);
  h5.createDataSet("earnings/diagnostics/selected_candidate",
                   selected_candidate);
  h5.createDataSet("earnings/diagnostics/gradient_unconstrained",
                   diagnostic_gradient);
  h5.createDataSet("earnings/diagnostics/move_unconstrained",
                   diagnostic_move);
  h5.createDataSet("earnings/diagnostics/transport_candidates/log_weight",
                   candidate_log_weight);
  h5.createDataSet("earnings/diagnostics/transport_candidates/probability",
                   candidate_probability);
  h5.createDataSet("earnings/diagnostics/transport_candidates/logp_gain",
                   candidate_logp_gain);
  h5.createDataSet("earnings/diagnostics/transport_candidates/jump_bonus",
                   candidate_jump_bonus);
  h5.createDataSet("earnings/diagnostics/transport_candidates/diag_jump",
                   candidate_diag_jump);
  h5.createDataSet("earnings/diagnostics/transport_candidates/move_norm",
                   candidate_move_norm);
  h5.createDataSet("earnings/diagnostics/transport_candidates/grad_dot_move",
                   candidate_grad_dot_move);
  h5.createDataSet("earnings/diagnostics/transport_candidates/cos_grad_move",
                   candidate_cos_grad_move);
  h5.createDataSet("earnings/diagnostics/transport_candidates/beta_slope",
                   candidate_beta_slope);
  h5.createDataSet("earnings/diagnostics/transport_candidates/delta_beta0",
                   candidate_delta_beta0);
  h5.createDataSet("earnings/diagnostics/transport_candidates/delta_beta1",
                   candidate_delta_beta1);

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';
  std::cout << "Output: " << config.output << '\n';

  return 0;
}
