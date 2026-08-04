// The collapsed Hastings ratio in BaseKLHR::kl_step_ is only valid because
// ordered overrelaxation is reversible with respect to the distribution it is
// applied to (Neal 1995, "Suppressing random walks in Markov chain Monte
// Carlo using ordered overrelaxation", section 4.2). These tests exercise
// that property directly on the u-space kernel, plus the pieces the ratio
// leans on: an exact CDF/quantile pair and correct KL gradients.

#include "klhr_numerics.hpp"
#include "normal_quantile.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

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

// Mirror of BaseKLHR::overrelaxed_proposal_impl_, standing alone so the
// kernel can be exercised without a Stan model.
class UniformOverrelaxation {
public:
  UniformOverrelaxation(const int K, const std::uint64_t seed) :
    K_(K), rng_(seed) {}

  double step(const double u) {
    if (K_ == 0 || klhr::numerics::probability_saturated(u)) {
      return klhr::numerics::clamp_probability(uniform_(rng_));
    }
    std::binomial_distribution<int> binomial(K_, u);
    const int r = binomial(rng_);
    double up = u;
    if (r > K_ - r) {
      up = u * beta_(K_ - r + 1.0, 2.0 * r - K_);
    } else if (r < K_ - r) {
      up = 1.0 - (1.0 - u) * beta_(r + 1.0, K_ - 2.0 * r);
    }
    return klhr::numerics::clamp_probability(up);
  }

private:
  double beta_(const double a, const double b) {
    std::gamma_distribution<double> ga(a, 1.0);
    std::gamma_distribution<double> gb(b, 1.0);
    double x = ga(rng_);
    double y = gb(rng_);
    double total = x + y;
    while (!std::isfinite(total) || total <= 0.0) {
      x = ga(rng_);
      y = gb(rng_);
      total = x + y;
    }
    return x / total;
  }

  int K_;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

// Chi-square goodness of fit against Uniform(0,1) over `bins` cells.
double uniformity_chi_square(const std::vector<double>& values,
                             const std::size_t bins) {
  std::vector<std::size_t> counts(bins, 0);
  for (const double v : values) {
    auto index = static_cast<std::size_t>(v * static_cast<double>(bins));
    counts[std::min(index, bins - 1)] += 1;
  }
  const double expected =
    static_cast<double>(values.size()) / static_cast<double>(bins);
  double total = 0.0;
  for (const std::size_t c : counts) {
    const double d = static_cast<double>(c) - expected;
    total += d * d / expected;
  }
  return total;
}

// One kernel step applied to independent Uniform(0,1) draws must come back
// Uniform(0,1). This is the invariance the cancellation in kl_step_ rests on.
// Testing one step from iid starts keeps the outputs independent, so the
// chi-square reference distribution is valid -- a long single chain would
// not work here, because overrelaxation deliberately makes successive draws
// strongly (and for large K, nearly antithetically) dependent.
void test_kernel_preserves_uniform() {
  // 99.9th percentile of chi-square with 19 degrees of freedom.
  constexpr double critical = 43.82;
  for (const int K : {0, 1, 3, 7, 15, 31, 63}) {
    UniformOverrelaxation kernel(K, 20240612u + static_cast<unsigned>(K));
    std::mt19937_64 start_rng(777u + static_cast<unsigned>(K));
    std::uniform_real_distribution<double> start(0.0, 1.0);

    std::vector<double> draws;
    draws.reserve(200000);
    for (int i = 0; i < 200000; ++i) {
      draws.push_back(kernel.step(start(start_rng)));
    }
    const double chi2 = uniformity_chi_square(draws, 20);
    check(chi2 < critical, "overrelaxation kernel preserves Uniform(0,1)",
          __LINE__);
    if (chi2 >= critical) {
      std::cerr << "    K=" << K << " chi2=" << chi2 << '\n';
    }
  }
}

// The kernel should also be antithetic: for K above 1 a draw below the
// median should tend to land above it, which is the whole point of
// overrelaxation. This guards against a kernel that is uniform-preserving
// but has silently become an independence sampler.
void test_kernel_is_antithetic() {
  std::mt19937_64 start_rng(4242u);
  std::uniform_real_distribution<double> start(0.0, 1.0);
  double previous_correlation = 1.0;
  for (const int K : {3, 7, 15, 31, 63}) {
    UniformOverrelaxation kernel(K, 31337u + static_cast<unsigned>(K));
    double sum_uv = 0.0;
    constexpr int n = 200000;
    for (int i = 0; i < n; ++i) {
      const double u = start(start_rng);
      const double v = kernel.step(u);
      sum_uv += (u - 0.5) * (v - 0.5);
    }
    // Var(U) = 1/12 for a uniform.
    const double correlation = (sum_uv / n) * 12.0;
    check(correlation < 0.0, "overrelaxation induces negative correlation",
          __LINE__);
    check(correlation < previous_correlation + 0.05,
          "larger K is at least as antithetic", __LINE__);
    if (!(correlation < 0.0)) {
      std::cerr << "    K=" << K << " corr=" << correlation << '\n';
    }
    previous_correlation = correlation;
  }
}

// Odd K only. An even K leaves a self-transition atom at the middle rank
// that is not part of the continuous kernel.
void test_middle_rank_atom_absent_for_odd_K() {
  for (const int K : {3, 7, 15, 31, 63}) {
    UniformOverrelaxation kernel(K, 99u + static_cast<unsigned>(K));
    std::size_t stayed = 0;
    double u = 0.37;
    for (int i = 0; i < 20000; ++i) {
      const double up = kernel.step(u);
      if (up == u) {
        ++stayed;
      }
      u = up;
    }
    CHECK(stayed == 0);
  }
}

// The kernel is applied in u-space, so F and its inverse have to be genuine
// inverses or the reversibility argument does not transfer back to x-space.
void test_quantile_inverts_cdf() {
  double worst = 0.0;
  for (int i = 1; i < 20000; ++i) {
    const double p = static_cast<double>(i) / 20000.0;
    const double z = klhr::normal_quantile(p);
    worst = std::max(worst, std::abs(klhr::normal_cdf(z) - p));
  }
  CHECK(worst < 1e-15);
  if (worst >= 1e-15) {
    std::cerr << "    worst |F(F^-1(p)) - p| = " << worst << '\n';
  }

  // Tail values, where the rational approximation is least accurate.
  for (const double p : {1e-8, 1e-5, 1e-3, 0.999, 1.0 - 1e-5, 1.0 - 1e-8}) {
    const double z = klhr::normal_quantile(p);
    CHECK(std::abs(klhr::normal_cdf(z) - p) <= 1e-14 * std::max(p, 1.0 - p));
  }
}

void test_bounded_reparam() {
  constexpr double radius = 5.0;
  CHECK(klhr::numerics::bounded_reparam(0.0, radius) == 0.0);
  for (const double raw : {-1e6, -3.0, -0.25, 0.5, 2.0, 1e6}) {
    const double value = klhr::numerics::bounded_reparam(raw, radius);
    // Saturates exactly at the radius once tanh rounds to one.
    CHECK(std::abs(value) <= radius);
    // Finite-difference the derivative.
    const double h = 1e-6;
    const double numeric =
      (klhr::numerics::bounded_reparam(raw + h, radius) -
       klhr::numerics::bounded_reparam(raw - h, radius)) / (2.0 * h);
    const double analytic =
      klhr::numerics::bounded_reparam_derivative(raw, radius);
    CHECK(std::abs(numeric - analytic) <= 1e-6 * std::max(1.0, analytic));
  }
  CHECK(klhr::numerics::bounded_reparam(
          std::numeric_limits<double>::quiet_NaN(), radius) == 0.0);
}

// The barrier has to pull back toward the feasible region rather than hand
// BFGS a flat plateau.
void test_bad_kl_barrier() {
  Eigen::VectorXd eta(3);
  eta << 2.0, -3.0, 0.5;
  double value = 0.0;
  Eigen::VectorXd grad;
  klhr::numerics::set_bad_kl(eta, value, grad);

  CHECK(klhr::numerics::is_bad_kl(value));
  CHECK(grad.size() == 3);
  // -grad is a descent direction back toward the origin in every component,
  // not just the scale.
  CHECK(grad.dot(eta) > 0.0);
  for (Eigen::Index i = 0; i < eta.size(); ++i) {
    CHECK(grad(i) == eta(i));
  }

  // Non-finite components must not leak into the gradient.
  Eigen::VectorXd broken(2);
  broken << std::numeric_limits<double>::quiet_NaN(), 1.5;
  double broken_value = 0.0;
  Eigen::VectorXd broken_grad;
  klhr::numerics::set_bad_kl(broken, broken_value, broken_grad);
  CHECK(broken_grad.allFinite());
  CHECK(broken_grad(0) == 0.0);
  CHECK(broken_grad(1) == 1.5);

  CHECK(!klhr::numerics::is_bad_kl(1.0));
  CHECK(!klhr::numerics::is_bad_kl(-1e50));
}

void test_probability_guards() {
  CHECK(klhr::numerics::probability_saturated(0.0));
  CHECK(klhr::numerics::probability_saturated(1.0));
  CHECK(klhr::numerics::probability_saturated(
          std::numeric_limits<double>::quiet_NaN()));
  CHECK(!klhr::numerics::probability_saturated(0.5));
  CHECK(!klhr::numerics::probability_saturated(1e-12));

  const double clamped =
    klhr::numerics::clamp_probability(std::numeric_limits<double>::infinity());
  CHECK(clamped == 0.5);
  CHECK(klhr::numerics::clamp_probability(2.0) < 1.0);
  CHECK(klhr::numerics::clamp_probability(-1.0) > 0.0);
}

} // namespace

int main() {
  test_kernel_preserves_uniform();
  test_kernel_is_antithetic();
  test_middle_rank_atom_absent_for_odd_K();
  test_quantile_inverts_cdf();
  test_bounded_reparam();
  test_bad_kl_barrier();
  test_probability_guards();
  return failures == 0 ? 0 : 1;
}
