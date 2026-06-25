#include "klhr.hpp"
#include "welford.hpp"

#include <Eigen/Dense>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>

#include <cstddef>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <utility>

int main() {

  constexpr std::size_t N = 30'000;
  constexpr std::size_t warmup = 15'000;
  std::string model_name = "earnings";
  std::string model = std::format("./stan/{}_model.so", model_name);
  std::string data = std::format("./stan/{}.json", model_name);
  klhr::KLHR algo{model, data, {.warmup = warmup}};
  std::size_t D = algo.dim();
  std::size_t C = algo.diagnostic_candidate_count_;
  WelfordAccumulator w{D};

  Eigen::MatrixXd draws(N, D);
  Eigen::VectorXd acceptance_rate(N);
  Eigen::VectorXd log_density(N);
  Eigen::VectorXd nfev(N);
  Eigen::VectorXd diagnostic_phase(N);
  Eigen::VectorXd grad_dot_move(N);
  Eigen::VectorXd cos_grad_move(N);
  Eigen::VectorXd beta_slope(N);
  Eigen::VectorXd diagnostic_logp_gain(N);
  Eigen::VectorXd jump_bonus(N);
  Eigen::VectorXd diag_jump(N);
  Eigen::VectorXd move_norm(N);
  Eigen::VectorXd grad_norm(N);
  Eigen::VectorXd selected_candidate(N);
  Eigen::MatrixXd diagnostic_gradient(N, D);
  Eigen::MatrixXd diagnostic_move(N, D);
  Eigen::MatrixXd candidate_log_weight(N, C);
  Eigen::MatrixXd candidate_probability(N, C);
  Eigen::MatrixXd candidate_logp_gain(N, C);
  Eigen::MatrixXd candidate_jump_bonus(N, C);
  Eigen::MatrixXd candidate_diag_jump(N, C);
  Eigen::MatrixXd candidate_move_norm(N, C);
  Eigen::MatrixXd candidate_grad_dot_move(N, C);
  Eigen::MatrixXd candidate_cos_grad_move(N, C);
  Eigen::MatrixXd candidate_beta_slope(N, C);
  Eigen::MatrixXd candidate_delta_beta0(N, C);
  Eigen::MatrixXd candidate_delta_beta1(N, C);

  std::filesystem::create_directories("draws");
  HighFive::File h5("draws/earnings.h5", HighFive::File::Truncate);

  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < N; ++n) {
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
    if (n >= warmup) {
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

  return 0;
}
