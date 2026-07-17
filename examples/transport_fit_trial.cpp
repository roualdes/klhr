#include "normal_klhr.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: transport_fit_trial SEED N MAXITER STEPS\n";
    return 2;
  }

  const std::uint64_t seed = std::stoull(argv[1]);
  const Eigen::Index nodes = std::stoll(argv[2]);
  const std::size_t maxiter = std::stoull(argv[3]);
  const std::size_t steps = std::stoull(argv[4]);
  if (seed == 0 || nodes < 1 || maxiter < 1 || steps < 1) {
    throw std::invalid_argument("all trial arguments must be positive");
  }

  klhr::KlhrOptions options{
    .seed = seed,
    .N = nodes,
    .transport_maxiter_bfgs = maxiter,
    .warmup = steps,
    .initial_transport_steps = steps,
  };
  klhr::NormalKLHR sampler("./stan/earnings_model.so",
                           "./stan/earnings.json", options);

  std::cout << "seed,nodes,maxiter,iteration,beta0,beta1,sigma,s,nfev,logp\n";
  std::cout << std::setprecision(17);
  for (std::size_t iteration = 1; iteration <= steps; ++iteration) {
    const Eigen::VectorXd draw = sampler.draw();
    if (draw.size() != 4) {
      throw std::runtime_error("earnings draw does not have four parameters");
    }
    std::cout << seed << ',' << nodes << ',' << maxiter << ','
              << iteration << ',' << draw(0) << ',' << draw(1) << ','
              << draw(2) << ',' << draw(3) << ',' << sampler.nfev_ << ','
              << sampler.log_density_ << '\n';
  }
}
