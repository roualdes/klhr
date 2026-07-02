#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace klhr {

namespace detail {

inline Eigen::VectorXd laguerre_n(const Eigen::VectorXd& x, std::size_t n) {
  if (n == 0) {
    return Eigen::VectorXd::Ones(x.size());
  }

  Eigen::VectorXd l0 = Eigen::VectorXd::Ones(x.size());
  Eigen::VectorXd l1 = Eigen::VectorXd::Ones(x.size()) - x;
  if (n == 1) {
    return l1;
  }

  for (std::size_t k = 1; k < n; ++k) {
    const double kd = static_cast<double>(k);
    Eigen::VectorXd l2 =
      ((2.0 * kd + 1.0) * l1 - x.cwiseProduct(l1) - kd * l0) /
      (kd + 1.0);
    l0 = l1;
    l1 = l2;
  }

  return l1;
}

inline Eigen::VectorXd laguerre_derivative_n(const Eigen::VectorXd& x,
                                             std::size_t n) {
  if (n == 0) {
    return Eigen::VectorXd::Zero(x.size());
  }

  const Eigen::VectorXd ln = laguerre_n(x, n);
  const Eigen::VectorXd lm = laguerre_n(x, n - 1);
  const double nd = static_cast<double>(n);
  return nd * (ln - lm).cwiseQuotient(x);
}

}  // namespace detail

inline void gauss_laguerre(std::size_t N, Eigen::VectorXd& ws,
                           Eigen::VectorXd& xs) {
  if (N == 0) {
    throw std::invalid_argument("gauss_laguerre: N must be positive");
  }

  if (N > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
    throw std::overflow_error("gauss_laguerre: N is too large for Eigen::Index");
  }

  const Eigen::Index n = static_cast<Eigen::Index>(N);
  ws.resize(n);
  xs.resize(n);

  Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    companion(i, i) = 2.0 * static_cast<double>(i) + 1.0;
  }
  for (Eigen::Index i = 1; i < n; ++i) {
    const double a = -static_cast<double>(i);
    companion(i - 1, i) = a;
    companion(i, i - 1) = a;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(companion,
                                                        Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("gauss_laguerre: eigenvalue computation failed");
  }

  Eigen::VectorXd x = solver.eigenvalues();  // sorted increasing

  // One Newton refinement step.
  Eigen::VectorXd dy = detail::laguerre_n(x, N);
  Eigen::VectorXd df = detail::laguerre_derivative_n(x, N);
  x -= dy.cwiseQuotient(df);

  // Compute weights, scaling factors to avoid possible overflow.
  Eigen::VectorXd fm = detail::laguerre_n(x, N - 1);
  df = detail::laguerre_derivative_n(x, N);

  const double fm_max = fm.cwiseAbs().maxCoeff();
  const double df_max = df.cwiseAbs().maxCoeff();
  if (!std::isfinite(fm_max) || !std::isfinite(df_max) ||
      fm_max == 0.0 || df_max == 0.0) {
    throw std::runtime_error("gauss_laguerre: weight computation failed");
  }

  fm /= fm_max;
  df /= df_max;
  Eigen::VectorXd w = fm.cwiseProduct(df).cwiseInverse();

  // Normalize so integral of 1 against exp(-x) is 1.
  w /= w.sum();

  xs = x;
  ws = w;
}

} // namespace klhr
