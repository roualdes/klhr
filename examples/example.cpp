#include "klhr.hpp"
#include "welford.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

int main() {
  klhr::KLHR algo{"./stan/earnings_model.so", "./stan/earnings.json"};
  std::size_t D = algo.dim();
  WelfordAccumulator w{D};

  std::size_t N = 30'000;
  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < N; ++n) {
    draw = algo.draw();
    w.update(draw);
  }

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "Number log_density evals: " << algo.nfev_ << '\n';

  return 0;
}
