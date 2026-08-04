// The KL objectives are evaluated by Gauss-Hermite (normal reference) and
// Gauss-Laguerre (Weibull reference) quadrature. A wrong node or weight
// silently biases every line fit, so check both rules against moments with
// known values, and check the rescaling the samplers apply on top.

#include "gausshermite.hpp"
#include "gausslaguerre.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <numbers>
#include <string_view>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view expression,
           const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: "
              << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

void close(const double actual, const double expected, const double tol,
           const std::string_view what, const int line) {
  const bool ok = std::abs(actual - expected) <= tol;
  if (!ok) {
    std::cerr << "line " << line << ": " << what << " got " << actual
              << " want " << expected << '\n';
    ++failures;
  }
}

#define CLOSE(actual, expected, tol) \
  close((actual), (expected), (tol), #actual, __LINE__)

// Gauss-Hermite rescaled the way BaseKLHR does: nodes * sqrt(2), weights
// / sqrt(pi), giving a rule for E[f(Z)] with Z standard normal.
void test_gauss_hermite_normal_moments() {
  for (const Eigen::Index n : {2, 4, 8, 16}) {
    Eigen::VectorXd w;
    Eigen::VectorXd x;
    klhr::gauss_hermite(n, w, x);
    CHECK(w.size() == n);
    CHECK(x.size() == n);
    x *= std::sqrt(2.0);
    w /= std::sqrt(std::numbers::pi_v<double>);

    CLOSE(w.sum(), 1.0, 1e-12);
    CLOSE(w.dot(x), 0.0, 1e-12);
    CLOSE(w.dot(x.cwiseProduct(x)), 1.0, 1e-12);
    if (n >= 4) {
      // E[Z^4] = 3
      const Eigen::VectorXd x4 =
        x.array().square().square().matrix();
      CLOSE(w.dot(x4), 3.0, 1e-10);
    }
    if (n >= 8) {
      // E[Z^6] = 15
      const Eigen::VectorXd x6 =
        (x.array().square() * x.array().square() * x.array().square())
        .matrix();
      CLOSE(w.dot(x6), 15.0, 1e-8);
    }
  }
}

// Gauss-Laguerre integrates against exp(-x) on (0, inf): E[X^k] = k! for
// X ~ Exponential(1).
void test_gauss_laguerre_exponential_moments() {
  for (const Eigen::Index n : {2, 4, 8, 16}) {
    Eigen::VectorXd w;
    Eigen::VectorXd x;
    klhr::gauss_laguerre(n, w, x);
    CHECK(w.size() == n);
    CHECK(x.size() == n);
    CHECK((x.array() > 0.0).all());

    CLOSE(w.sum(), 1.0, 1e-12);
    CLOSE(w.dot(x), 1.0, 1e-10);
    CLOSE(w.dot(x.cwiseProduct(x)), 2.0, 1e-9);
    if (n >= 4) {
      const Eigen::VectorXd x3 =
        (x.array() * x.array() * x.array()).matrix();
      CLOSE(w.dot(x3), 6.0, 1e-7);
    }
  }
}

// gauss_hermite must size its outputs itself; callers that pass empty
// vectors previously wrote out of bounds.
void test_outputs_are_resized() {
  Eigen::VectorXd w;
  Eigen::VectorXd x;
  klhr::gauss_hermite(8, w, x);
  CHECK(w.size() == 8);
  CHECK(x.size() == 8);

  Eigen::VectorXd w2(3);
  Eigen::VectorXd x2(3);
  klhr::gauss_hermite(8, w2, x2);
  CHECK(w2.size() == 8);
  CHECK(x2.size() == 8);
}

void test_nodes_are_symmetric() {
  Eigen::VectorXd w;
  Eigen::VectorXd x;
  klhr::gauss_hermite(8, w, x);
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const Eigen::Index j = x.size() - 1 - i;
    CLOSE(x(i), -x(j), 1e-12);
    CLOSE(w(i), w(j), 1e-12);
  }
}

} // namespace

int main() {
  test_gauss_hermite_normal_moments();
  test_gauss_laguerre_exponential_moments();
  test_outputs_are_resized();
  test_nodes_are_symmetric();
  return failures == 0 ? 0 : 1;
}
