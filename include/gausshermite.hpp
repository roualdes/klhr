#include <Eigen/Dense>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace klhr {

namespace detail {

Eigen::VectorXd normed_hermite_n(const Eigen::VectorXd& x,
                                 const Eigen::Index n) {
  const double pi = std::numbers::pi_v<double>;
  const double pi_mquarter = 1.0 / std::sqrt(std::sqrt(pi));

  if (n == 0) {
    return Eigen::VectorXd::Constant(x.size(), pi_mquarter);
  }

  Eigen::VectorXd c0 = Eigen::VectorXd::Zero(x.size());
  Eigen::VectorXd c1 = Eigen::VectorXd::Constant(x.size(), pi_mquarter);

  double nd = n;

  for (Eigen::Index i = 0; i + 1 < n; ++i) {
    Eigen::VectorXd tmp = c0;
    c0 = -c1 * std::sqrt((nd - 1.0) / nd);
    c1 = tmp + c1.cwiseProduct(x) * std::sqrt(2.0 / nd);
    nd -= 1.0;
  }

  return c0 + c1.cwiseProduct(x) * std::sqrt(2.0);
}

}  // namespace detail

void gauss_hermite(const Eigen::Index n, Eigen::VectorXd& ws,
                   Eigen::VectorXd& xs) {
  if (n <= 0) {
    throw std::invalid_argument("gauss_hermite: N must be positive");
  }

  const double pi = std::numbers::pi_v<double>;

  Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(n, n);
  for (Eigen::Index i = 1; i < n; ++i) {
    const double a = std::sqrt(0.5 * i);
    companion(i - 1, i) = a;
    companion(i, i - 1) = a;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(companion, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("gauss_hermite: eigenvalue computation failed");
  }

  Eigen::VectorXd x = solver.eigenvalues();  // sorted increasing, like eigvalsh

  // One Newton refinement step:
  // x -= H_n(x) / H'_n(x), using normalized Hermite recurrence.
  Eigen::VectorXd dy = detail::normed_hermite_n(x, n);
  Eigen::VectorXd df =
    detail::normed_hermite_n(x, n - 1) * std::sqrt(2.0 * n);

  x -= dy.cwiseQuotient(df);

  // Compute weights, scaling the factor to reduce overflow risk.
  Eigen::VectorXd fm = detail::normed_hermite_n(x, n - 1);
  const double fm_max = fm.cwiseAbs().maxCoeff();
  fm /= fm_max;

  Eigen::VectorXd w = fm.array().square().inverse().matrix();

  // symmetrize
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::Index j = n - 1 - i;
    ws(i) = 0.5 * (w(i) + w(j));
    xs(i) = 0.5 * (x(i) - x(j));
  }

  // Normalize so integral of 1 against exp(-x^2) is sqrt(pi).
  ws *= std::sqrt(pi) / ws.sum();
}

} // namespace klhr
