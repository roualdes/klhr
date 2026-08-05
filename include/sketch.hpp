#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace klhr {

// Single-pass Nystrom sketch of the covariance of the centered draws.
//
// A fixed Gaussian probe matrix Omega (D by m) is drawn once, and the sketch
// accumulates
//
//     Y = sum_i x_i (x_i' Omega) = S Omega,      S the scatter matrix
//
// one rank-one update per draw. At a window close the m-by-m matrix Omega'Y =
// Omega' S Omega is formed and the Nystrom approximation
//
//     S  ~  Y (Omega' S Omega)^+ Y'
//
// gives a square root F with F F' approximating the covariance, using every
// draw in the window at O(D m) memory and O(D m) work per draw. Nothing
// D-by-D is ever formed.
//
// Used only by the mixture's SKETCH component, which needs a square root to
// sample directions from. It was also trialled as a drop-in replacement for
// the incremental OnlinePCA basis and made no difference: across four models,
// four ranks and two error metrics, not one comparison resolved away from
// parity. OnlinePCA was never the bottleneck for the covariance direction law,
// so that path was removed.
//
// (An earlier ring-buffer version, holding only the last m states with m
// clamped at D, did lose to OnlinePCA by 2.3x on corr-normal at J=10 -- but
// that was sample starvation, not the estimator. Hence the streaming form
// here, which uses every draw in the window.)
//
// The buffer holds centered states rather than lagged differences on purpose.
// For a stationary chain the covariance of lag-L differences is
// 2 (Sigma - Sigma_L), whose eigenstructure suppresses exactly the slowly
// mixing directions a direction law most needs to propose along. The scale
// would be harmless -- it is absorbed when rho is normalised -- the shape is
// not.
class Sketch {
public:
  Sketch(Eigen::Index D = 0, Eigen::Index m = 0, double tol = 1e-10,
         std::uint64_t seed = 1) :
    D_(checked_dimension_(D)),
    m_(checked_dimension_(m)),
    tol_(tol),
    Omega_(Eigen::MatrixXd::Zero(D_, std::max<Eigen::Index>(m_, 1))),
    Y_(Eigen::MatrixXd::Zero(D_, std::max<Eigen::Index>(m_, 1))),
    count_(0) {
    if (m_ > 0 && D_ > 0) {
      // The probes are fixed for the life of the run: re-drawing them each
      // window would make successive bases incomparable for no benefit.
      std::mt19937_64 gen(seed == 0 ? 1 : seed);
      std::normal_distribution<double> normal(0.0, 1.0);
      for (Eigen::Index j = 0; j < m_; ++j) {
        for (Eigen::Index i = 0; i < D_; ++i) {
          Omega_(i, j) = normal(gen);
        }
      }
    }
  }

  void update(const Eigen::Ref<const Eigen::VectorXd>& centered) {
    if (m_ <= 0) {
      return;
    }
    if (centered.size() != D_) {
      throw std::invalid_argument("Sketch::update: input dimension mismatch");
    }
    if (!centered.allFinite()) {
      return;
    }
    // Y += x (x' Omega): one rank-one update, O(D m).
    Y_.noalias() += centered * (centered.transpose() * Omega_);
    ++count_;
  }

  // Remove the bias from having centred on a stale mean.
  //
  // With x_i = u_i + delta, where u_i is centred on the window's own mean and
  // delta is the shift between that and the mean actually used,
  //
  //   Y = sum_i x_i (x_i' Omega)
  //     = Y_true + (sum_i u_i)(delta' Omega) + delta (sum_i u_i)' Omega
  //       + n delta (delta' Omega)
  //     = Y_true + n delta (delta' Omega),
  //
  // since sum_i u_i vanishes. The correction is therefore exact, not an
  // approximation, and costs one rank-one update.
  //
  // Without it the scatter is inflated by the square of the within-window
  // drift, which is why a rank-20 subspace was reporting up to 188% of the
  // total variance -- impossible for any correct estimator, and worst on the
  // slowest-mixing high-dimensional targets.
  void recenter(const Eigen::Ref<const Eigen::VectorXd>& delta) {
    if (m_ <= 0 || count_ <= 0 || delta.size() != D_ || !delta.allFinite()) {
      return;
    }
    Y_.noalias() -= static_cast<double>(count_) *
      (delta * (delta.transpose() * Omega_));
  }

  void reset() {
    Y_.setZero();
    count_ = 0;
  }

  // The sketch as a probe/image pair source for the fused fit: the probes are
  // Omega and the images are Y / count, since E[Y_j] = n Sigma Omega_j. Two
  // sketches constructed with the same seed and dimensions share Omega
  // exactly, which is what lets a window be split into halves whose images are
  // directly comparable.
  const Eigen::MatrixXd& probes() const { return Omega_; }

  bool mean_image(Eigen::MatrixXd& out) const {
    if (m_ <= 0 || count_ <= 0) {
      return false;
    }
    out = Y_ / static_cast<double>(count_);
    return out.allFinite();
  }

  Eigen::Index count() const { return count_; }

  // Square root B with B B' = sum_j sigma2_j u_j u_j', the sketched
  // covariance.
  //
  // Spectral shrinkage -- subtracting a multiple of the median eigenvalue from
  // each, so that an all-bulk spectrum collapses and the component drops out
  // on its own -- was added here as a structure test and measured not to be
  // one. At ESS/D between 0.09 and 0.26 on these targets the spectrum is
  // noise-dominated whatever the truth is, so ar1 (strongly correlated) and
  // ill-normal (not correlated at all) kept indistinguishable rank at every
  // threshold tried. Removed; the rank cap is the only knob left.
  bool factor(Eigen::MatrixXd& B,
              const Eigen::Index max_rank = 0) const {
    Eigen::MatrixXd F;
    Eigen::MatrixXd U;
    Eigen::VectorXd sigma2;
    if (!build_factor_(F) || !spectrum_(F, U, sigma2)) {
      return false;
    }
    const Eigen::Index k = sigma2.size();
    // spectrum_ returns eigenvalues in descending order.
    Eigen::Index rank = 0;
    while (rank < k && sigma2(rank) > 0.0) {
      ++rank;
    }
    if (max_rank > 0) {
      rank = std::min(rank, max_rank);
    }
    if (rank <= 0) {
      return false;
    }
    B.resize(D_, rank);
    for (Eigen::Index j = 0; j < rank; ++j) {
      B.col(j) = std::sqrt(sigma2(j)) * U.col(j);
    }
    return B.allFinite();
  }

private:
  // F with F F' ~ covariance, from the Nystrom identity. Eigendecomposing
  // Omega'Y rather than taking its Cholesky keeps this well defined while the
  // window is still short and the scatter is rank deficient.
  bool build_factor_(Eigen::MatrixXd& F) const {
    if (m_ <= 0 || count_ <= 1) {
      return false;
    }
    Eigen::MatrixXd core = Omega_.transpose() * Y_;
    core = (0.5 * (core + core.transpose())).eval();
    if (!core.allFinite()) {
      return false;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(core);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    const Eigen::VectorXd& lambda = solver.eigenvalues();
    const double lambda_max = lambda(m_ - 1);
    if (!std::isfinite(lambda_max) || lambda_max <= tol_) {
      return false;
    }

    Eigen::Index keep = 0;
    while (keep < m_ && lambda(m_ - 1 - keep) > 1e-10 * lambda_max) {
      ++keep;
    }
    if (keep <= 0) {
      return false;
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(count_));
    F.resize(D_, keep);
    for (Eigen::Index j = 0; j < keep; ++j) {
      const Eigen::Index source = m_ - 1 - j;
      F.col(j) = scale * (Y_ * solver.eigenvectors().col(source)) /
        std::sqrt(lambda(source));
    }
    return F.allFinite();
  }

  // Left singular vectors and squared singular values of F, recovered from
  // the small Gram F'F so that nothing D-by-D is decomposed.
  bool spectrum_(const Eigen::MatrixXd& F, Eigen::MatrixXd& U,
                 Eigen::VectorXd& sigma2) const {
    Eigen::MatrixXd G = F.transpose() * F;
    G = (0.5 * (G + G.transpose())).eval();
    if (!G.allFinite()) {
      return false;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(G);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    const Eigen::VectorXd& values = solver.eigenvalues();
    const Eigen::Index k = values.size();
    const double value_max = values(k - 1);
    if (!std::isfinite(value_max) || value_max <= tol_) {
      return false;
    }

    Eigen::Index keep = 0;
    while (keep < k && values(k - 1 - keep) > 1e-12 * value_max) {
      ++keep;
    }
    if (keep <= 0) {
      return false;
    }

    U.resize(D_, keep);
    sigma2.resize(keep);
    for (Eigen::Index j = 0; j < keep; ++j) {
      const Eigen::Index source = k - 1 - j;
      const double value = values(source);
      U.col(j) = (F * solver.eigenvectors().col(source)) / std::sqrt(value);
      sigma2(j) = value;
    }
    return U.allFinite() && sigma2.allFinite();
  }

  static Eigen::Index checked_dimension_(const Eigen::Index value) {
    if (value < 0) {
      throw std::invalid_argument("Sketch: dimensions must be nonnegative");
    }
    return value;
  }

  Eigen::Index D_;
  Eigen::Index m_;
  double tol_;
  Eigen::MatrixXd Omega_;
  Eigen::MatrixXd Y_;
  Eigen::Index count_;
};

} // namespace klhr
