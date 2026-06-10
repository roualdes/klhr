#include "klhr.hpp"
#include "welford.hpp"

#include <cstddef>
#include <iostream>

int main() {
  klhr::KLHR algo{"./stan/garch_model.so", "./stan/garch.json", {.stepsize = 0.33}};
  std::size_t D = algo.dim();
  WelfordAccumulator w{D};

  std::size_t N = 100'000;
  Eigen::VectorXd draw(D);
  for (std::size_t n = 0; n < N; ++n) {
    draw = algo.draw();
    w.update(draw);
  }

  std::cout << "Means: " << std::endl;
  std::cout << w.mean() << std::endl;
  std::cout << "Vars: " << std::endl;
  std::cout << w.variance() << std::endl;

  return 0;
}
