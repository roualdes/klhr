#include "klhr.hpp"
#include "welford.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

int main() {
  constexpr std::size_t warmup = 15'000;
  klhr::KLHR algo{"./stan/earnings_model.so", "./stan/earnings.json",
                  {.warmup = warmup}};
  std::size_t D = algo.dim();
  WelfordAccumulator w{D};

  std::size_t N = 30'000;
  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < N; ++n) {
    draw = algo.draw();
    if (n >= warmup) {
      w.update(draw);
    }
  }

  std::cout << "means: " << w.mean().transpose() << '\n';
  std::cout << "stds: " << w.std().transpose() << '\n';
  std::cout << "Number log_density evals: " << algo.nfev_ << '\n';
  std::cout << "Acceptance rate: " << algo.acceptance_rate_ << '\n';

  return 0;
}
