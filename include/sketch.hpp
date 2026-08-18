#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>

namespace klhr {

// A single-pass Nystrom estimate of a sample covariance. The sketch owns its
// centering, so callers pass raw positions.
class CovarianceSketch {
public:
  CovarianceSketch(Eigen::Index dimension = 0,
                   Eigen::Index columns = 0,
                   double tolerance = 1e-10,
                   std::uint64_t seed = 1) :
    dimension_(checked_dimension_(dimension)),
    columns_(checked_dimension_(columns)),
    tolerance_(tolerance),
    probes_(Eigen::MatrixXd::Zero(
      dimension_, std::max<Eigen::Index>(columns_, 1))),
    image_(Eigen::MatrixXd::Zero(
      dimension_, std::max<Eigen::Index>(columns_, 1))),
    sum_(Eigen::VectorXd::Zero(dimension_)) {
    if (columns_ > 0 && dimension_ > 0) {
      std::mt19937_64 generator(seed == 0 ? 1 : seed);
      std::normal_distribution<double> normal(0.0, 1.0);
      for (Eigen::Index j = 0; j < columns_; ++j) {
        for (Eigen::Index i = 0; i < dimension_; ++i) {
          probes_(i, j) = normal(generator);
        }
      }
    }
  }

  void update(const Eigen::Ref<const Eigen::VectorXd>& sample) {
    if (columns_ <= 0) {
      return;
    }
    if (sample.size() != dimension_) {
      throw std::invalid_argument(
        "CovarianceSketch::update: input dimension mismatch");
    }
    if (!sample.allFinite()) {
      return;
    }
    image_.noalias() += sample * (sample.transpose() * probes_);
    sum_ += sample;
    ++count_;
  }

  void reset() {
    image_.setZero();
    sum_.setZero();
    count_ = 0;
  }

  Eigen::Index count() const { return count_; }

  // Return leading eigenpairs of
  //
  //   diag(row_scale) Cov(sample) diag(row_scale).
  //
  // The covariance is first approximated in its raw coordinates, then the
  // small factor is row-scaled and diagonalised.
  bool eigenpairs(const Eigen::Ref<const Eigen::VectorXd>& row_scale,
                  Eigen::MatrixXd& directions,
                  Eigen::VectorXd& eigenvalues,
                  const Eigen::Index maximum_rank,
                  const double eigenvalue_cutoff) const {
    directions.resize(0, 0);
    eigenvalues.resize(0);
    if (row_scale.size() != dimension_ || !row_scale.allFinite() ||
        (row_scale.array() <= 0.0).any()) {
      return false;
    }

    Eigen::MatrixXd factor;
    if (!build_factor_(factor)) {
      return false;
    }
    factor.array().colwise() *= row_scale.array();

    Eigen::MatrixXd all_directions;
    Eigen::VectorXd all_values;
    if (!spectrum_(factor, all_directions, all_values)) {
      return false;
    }

    Eigen::Index rank = 0;
    while (rank < all_values.size() &&
           all_values(rank) >= eigenvalue_cutoff) {
      ++rank;
    }
    if (maximum_rank > 0) {
      rank = std::min(rank, maximum_rank);
    }
    if (rank <= 0) {
      return false;
    }

    directions = all_directions.leftCols(rank);
    eigenvalues = all_values.head(rank);
    return directions.allFinite() && eigenvalues.allFinite();
  }

private:
  // If s = sum_i x_i, the centered scatter image is
  //
  //   (sum_i x_i x_i' - s s' / n) Omega.
  //
  // Its Nystrom approximation is factored without forming a D-by-D matrix.
  bool build_factor_(Eigen::MatrixXd& factor) const {
    if (columns_ <= 0 || count_ <= 1) {
      return false;
    }

    Eigen::MatrixXd centered_image = image_;
    centered_image.noalias() -=
      (sum_ / static_cast<double>(count_)) * (sum_.transpose() * probes_);

    Eigen::MatrixXd core = probes_.transpose() * centered_image;
    core = (0.5 * (core + core.transpose())).eval();
    if (!core.allFinite()) {
      return false;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(core);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    const Eigen::VectorXd& values = solver.eigenvalues();
    const double maximum = values(columns_ - 1);
    if (!std::isfinite(maximum) || maximum <= tolerance_) {
      return false;
    }

    Eigen::Index keep = 0;
    while (keep < columns_ &&
           values(columns_ - 1 - keep) > 1e-10 * maximum) {
      ++keep;
    }
    if (keep <= 0) {
      return false;
    }

    factor.resize(dimension_, keep);
    const double covariance_scale =
      1.0 / std::sqrt(static_cast<double>(count_));
    for (Eigen::Index j = 0; j < keep; ++j) {
      const Eigen::Index source = columns_ - 1 - j;
      factor.col(j) = covariance_scale *
        (centered_image * solver.eigenvectors().col(source)) /
        std::sqrt(values(source));
    }
    return factor.allFinite();
  }

  static bool spectrum_(const Eigen::MatrixXd& factor,
                        Eigen::MatrixXd& directions,
                        Eigen::VectorXd& eigenvalues) {
    Eigen::MatrixXd gram = factor.transpose() * factor;
    gram = (0.5 * (gram + gram.transpose())).eval();
    if (!gram.allFinite()) {
      return false;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram);
    if (solver.info() != Eigen::Success) {
      return false;
    }
    const Eigen::VectorXd& values = solver.eigenvalues();
    const Eigen::Index size = values.size();
    const double maximum = values(size - 1);
    if (!std::isfinite(maximum) || maximum <= 0.0) {
      return false;
    }

    Eigen::Index keep = 0;
    while (keep < size && values(size - 1 - keep) > 1e-12 * maximum) {
      ++keep;
    }
    if (keep <= 0) {
      return false;
    }

    directions.resize(factor.rows(), keep);
    eigenvalues.resize(keep);
    for (Eigen::Index j = 0; j < keep; ++j) {
      const Eigen::Index source = size - 1 - j;
      const double value = values(source);
      directions.col(j) =
        (factor * solver.eigenvectors().col(source)) / std::sqrt(value);
      eigenvalues(j) = value;
    }
    return directions.allFinite() && eigenvalues.allFinite();
  }

  static Eigen::Index checked_dimension_(const Eigen::Index value) {
    if (value < 0) {
      throw std::invalid_argument(
        "CovarianceSketch: dimensions must be nonnegative");
    }
    return value;
  }

  Eigen::Index dimension_;
  Eigen::Index columns_;
  double tolerance_;
  Eigen::MatrixXd probes_;
  Eigen::MatrixXd image_;
  Eigen::VectorXd sum_;
  Eigen::Index count_ = 0;
};

// Common spectral direction law. Estimators only have to supply orthonormal
// directions and their covariance eigenvalues; rank diagnostics, trace
// allocation and drawing are identical after that point.
class TraceControlledDirection {
public:
  void begin_refresh() {
    ready_ = false;
    directions_.resize(0, 0);
    square_root_spectrum_.resize(0);
    spectrum_sum_ = 0.0;
  }

  bool install(Eigen::MatrixXd directions,
               Eigen::VectorXd covariance_eigenvalues) {
    if (directions.cols() <= 0 ||
        directions.cols() != covariance_eigenvalues.size() ||
        !directions.allFinite() || !covariance_eigenvalues.allFinite() ||
        (covariance_eigenvalues.array() <= 0.0).any()) {
      finish_refresh_(false);
      return false;
    }
    directions_ = std::move(directions);
    spectrum_sum_ = covariance_eigenvalues.sum();
    square_root_spectrum_ = covariance_eigenvalues.array().sqrt().matrix();
    const bool valid = square_root_spectrum_.allFinite() &&
      std::isfinite(spectrum_sum_) && spectrum_sum_ > 0.0;
    finish_refresh_(valid);
    return valid;
  }

  Eigen::VectorXd transform(
      Eigen::VectorXd full_rank_standard_normal,
      const Eigen::Ref<const Eigen::VectorXd>& subspace_standard_normal,
      const double residual_fraction) const {
    if (!ready_) {
      return full_rank_standard_normal;
    }
    if (full_rank_standard_normal.size() != directions_.rows() ||
        subspace_standard_normal.size() != directions_.cols()) {
      throw std::invalid_argument(
        "TraceControlledDirection::transform: input dimension mismatch");
    }

    const double residual = std::clamp(residual_fraction, 0.0, 1.0);
    const double subspace_scale = std::sqrt(
      (1.0 - residual) * static_cast<double>(directions_.rows()) /
      spectrum_sum_);
    full_rank_standard_normal *= std::sqrt(residual);
    full_rank_standard_normal.noalias() += subspace_scale * directions_ *
      square_root_spectrum_.cwiseProduct(subspace_standard_normal);
    return full_rank_standard_normal;
  }

  Eigen::VectorXd pure_transform(
      const Eigen::Ref<const Eigen::VectorXd>& subspace_standard_normal) const {
    if (!ready_) {
      return Eigen::VectorXd::Zero(directions_.rows());
    }
    if (subspace_standard_normal.size() != directions_.cols()) {
      throw std::invalid_argument(
        "TraceControlledDirection::pure_transform: input dimension mismatch");
    }

    const double subspace_scale = std::sqrt(
      static_cast<double>(directions_.rows()) / spectrum_sum_);
    return subspace_scale * directions_ *
      square_root_spectrum_.cwiseProduct(subspace_standard_normal);
  }

  void reset() {
    directions_.resize(0, 0);
    square_root_spectrum_.resize(0);
    spectrum_sum_ = 0.0;
    ready_ = false;
    last_rank_ = 0;
    refreshes_ = 0;
    rank_total_ = 0;
    dropouts_ = 0;
  }

  bool ready() const { return ready_; }
  Eigen::Index rank() const { return last_rank_; }
  std::size_t dropouts() const { return dropouts_; }
  double mean_rank() const {
    return refreshes_ == 0 ? 0.0 :
      static_cast<double>(rank_total_) / static_cast<double>(refreshes_);
  }

private:
  void finish_refresh_(const bool valid) {
    ready_ = valid;
    last_rank_ = ready_ ? directions_.cols() : 0;
    ++refreshes_;
    rank_total_ += static_cast<std::size_t>(last_rank_);
    if (!ready_) {
      directions_.resize(0, 0);
      square_root_spectrum_.resize(0);
      spectrum_sum_ = 0.0;
      ++dropouts_;
    }
  }

  Eigen::MatrixXd directions_;
  Eigen::VectorXd square_root_spectrum_;
  double spectrum_sum_ = 0.0;
  bool ready_ = false;
  Eigen::Index last_rank_ = 0;
  std::size_t refreshes_ = 0;
  std::size_t rank_total_ = 0;
  std::size_t dropouts_ = 0;
};

// Position samples estimate broad covariance eigenpairs directly.
class PositionSketch {
public:
  PositionSketch(Eigen::Index dimension,
                 Eigen::Index columns,
                 double tolerance,
                 std::uint64_t seed) :
    covariance_(dimension, columns, tolerance, seed) {}

  void update(const Eigen::Ref<const Eigen::VectorXd>& position) {
    covariance_.update(position);
  }

  void refresh(const Eigen::Ref<const Eigen::VectorXd>& metric_scale,
               const Eigen::Index maximum_rank,
               const double eigenvalue_cutoff) {
    direction_.begin_refresh();
    Eigen::MatrixXd directions;
    Eigen::VectorXd eigenvalues;
    const Eigen::VectorXd row_scale = metric_scale.cwiseInverse();
    const bool ready = covariance_.eigenpairs(
      row_scale, directions, eigenvalues, maximum_rank, eigenvalue_cutoff);
    covariance_.reset();
    if (ready) {
      direction_.install(std::move(directions), std::move(eigenvalues));
    } else {
      direction_.install(Eigen::MatrixXd{}, Eigen::VectorXd{});
    }
  }

  Eigen::VectorXd transform(
      Eigen::VectorXd full_rank_standard_normal,
      const Eigen::Ref<const Eigen::VectorXd>& subspace_standard_normal,
      const double residual_fraction) const {
    return direction_.transform(std::move(full_rank_standard_normal),
                                subspace_standard_normal,
                                residual_fraction);
  }

  void reset() { covariance_.reset(); direction_.reset(); }
  bool ready() const { return direction_.ready(); }
  Eigen::Index rank() const { return direction_.rank(); }
  std::size_t dropouts() const { return direction_.dropouts(); }
  double mean_rank() const { return direction_.mean_rank(); }

private:
  CovarianceSketch covariance_;
  TraceControlledDirection direction_;
};

// Direct local curvature in standardized coordinates. The caller supplies the
// negative log-density Hessian H and diagonal metric scale S. We diagonalise
// A = S H S, then use the broad covariance modes (the smallest eigenvalues of
// A, inverted) as a pure low-rank direction law. A non-positive local Hessian
// disables the component for that window rather than silently repairing it.
class CurvatureDirection {
public:
  CurvatureDirection(Eigen::Index dimension = 0,
                     double tolerance = 1e-10) :
    dimension_(checked_dimension_(dimension)),
    tolerance_(tolerance) {}

  bool refresh(const Eigen::Ref<const Eigen::MatrixXd>& negative_hessian,
               const Eigen::Ref<const Eigen::VectorXd>& metric_scale,
               const Eigen::Index maximum_rank,
               const double covariance_eigenvalue_cutoff) {
    direction_.begin_refresh();
    if (dimension_ <= 0 || negative_hessian.rows() != dimension_ ||
        negative_hessian.cols() != dimension_ ||
        metric_scale.size() != dimension_ ||
        !negative_hessian.allFinite() || !metric_scale.allFinite() ||
        (metric_scale.array() <= 0.0).any()) {
      return fail_();
    }

    Eigen::MatrixXd standardized = metric_scale.asDiagonal() *
      negative_hessian * metric_scale.asDiagonal();
    standardized =
      (0.5 * (standardized + standardized.transpose())).eval();
    if (!standardized.allFinite()) {
      return fail_();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(standardized);
    if (solver.info() != Eigen::Success) {
      return fail_();
    }

    const Eigen::VectorXd& curvatures = solver.eigenvalues();
    const double scale = std::max(1.0, curvatures.cwiseAbs().maxCoeff());
    const double positive_tolerance = tolerance_ * scale;
    if (!curvatures.allFinite() ||
        (curvatures.array() <= positive_tolerance).any()) {
      return fail_();
    }

    const Eigen::Index rank = std::clamp(
      maximum_rank, Eigen::Index{1}, dimension_);
    Eigen::Index retained = 0;
    while (retained < dimension_ && retained < rank) {
      if (1.0 / curvatures(retained) < covariance_eigenvalue_cutoff) {
        break;
      }
      ++retained;
    }
    if (retained <= 0) {
      return fail_();
    }

    return direction_.install(
      solver.eigenvectors().leftCols(retained),
      curvatures.head(retained).cwiseInverse());
  }

  Eigen::VectorXd transform(
      const Eigen::Ref<const Eigen::VectorXd>& subspace_standard_normal) const {
    if (!ready()) {
      return Eigen::VectorXd::Zero(dimension_);
    }
    return direction_.pure_transform(subspace_standard_normal);
  }

  void reset() { direction_.reset(); }
  bool ready() const { return direction_.ready(); }
  Eigen::Index rank() const { return direction_.rank(); }
  std::size_t dropouts() const { return direction_.dropouts(); }
  double mean_rank() const { return direction_.mean_rank(); }

private:
  bool fail_() {
    direction_.install(Eigen::MatrixXd{}, Eigen::VectorXd{});
    return false;
  }

  static Eigen::Index checked_dimension_(const Eigen::Index value) {
    if (value < 0) {
      throw std::invalid_argument(
        "CurvatureDirection: dimensions must be nonnegative");
    }
    return value;
  }

  Eigen::Index dimension_;
  double tolerance_;
  TraceControlledDirection direction_;
};

} // namespace klhr
