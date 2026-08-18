#include "sketch.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
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

bool near(const double actual, const double expected,
          const double tolerance = 1e-8) {
  return std::abs(actual - expected) <= tolerance;
}

void add_three_dimensional_samples(klhr::PositionSketch& sketch) {
  // Six zero-mean samples with maximum-likelihood covariance diag(9, 4, 1).
  sketch.update((Eigen::Vector3d() << std::sqrt(27.0), 0.0, 0.0).finished());
  sketch.update((Eigen::Vector3d() << -std::sqrt(27.0), 0.0, 0.0).finished());
  sketch.update((Eigen::Vector3d() << 0.0, std::sqrt(12.0), 0.0).finished());
  sketch.update((Eigen::Vector3d() << 0.0, -std::sqrt(12.0), 0.0).finished());
  sketch.update((Eigen::Vector3d() << 0.0, 0.0, std::sqrt(3.0)).finished());
  sketch.update((Eigen::Vector3d() << 0.0, 0.0, -std::sqrt(3.0)).finished());
}

void test_internal_centering_and_cutoff() {
  klhr::CovarianceSketch sketch(2, 2, 1e-12, 17);
  const Eigen::Vector2d shift(1000.0, -300.0);
  sketch.update(shift + Eigen::Vector2d(std::sqrt(8.0), 0.0));
  sketch.update(shift + Eigen::Vector2d(-std::sqrt(8.0), 0.0));
  sketch.update(shift + Eigen::Vector2d(0.0, std::sqrt(2.0)));
  sketch.update(shift + Eigen::Vector2d(0.0, -std::sqrt(2.0)));

  Eigen::MatrixXd directions;
  Eigen::VectorXd eigenvalues;
  CHECK(sketch.eigenpairs(Eigen::Vector2d::Ones(), directions, eigenvalues,
                          10, 2.0));
  CHECK(eigenvalues.size() == 1);
  CHECK(near(eigenvalues(0), 4.0, 1e-7));
  CHECK(near(std::abs(directions(0, 0)), 1.0, 1e-7));
  CHECK(near(directions(1, 0), 0.0, 1e-7));

  CHECK(!sketch.eigenpairs(Eigen::Vector2d::Ones(), directions, eigenvalues,
                           10, 5.0));
}

void test_position_trace_controlled_draw() {
  klhr::PositionSketch position(3, 3, 1e-12, 29);
  add_three_dimensional_samples(position);

  const Eigen::Vector3d metric_scale = Eigen::Vector3d::Ones();
  position.refresh(metric_scale, 10, 2.0);

  CHECK(position.ready());
  CHECK(position.rank() == 2);

  const Eigen::Vector3d full_rank_noise = Eigen::Vector3d::Ones();
  const Eigen::Vector2d subspace_noise = Eigen::Vector2d::Ones();
  constexpr double residual_fraction = 0.2;
  const Eigen::VectorXd position_output = position.transform(
    full_rank_noise, subspace_noise, residual_fraction);

  const double residual = std::sqrt(residual_fraction);
  // The unselected coordinate contains only the full-rank residual.
  CHECK(near(position_output(2), residual));
  // Trace normalization preserves the relative spectral weight sqrt(9/4).
  const double position_ratio =
    std::abs(position_output(0) - residual) /
    std::abs(position_output(1) - residual);
  CHECK(near(position_ratio, 1.5));
}

void test_common_rank_cap() {
  klhr::PositionSketch sketch(3, 3, 1e-12, 41);
  add_three_dimensional_samples(sketch);

  sketch.refresh(Eigen::Vector3d::Ones(), 1, 2.0);
  CHECK(sketch.ready());
  CHECK(sketch.rank() == 1);
}

void test_exact_curvature_finds_broad_mode() {
  klhr::CurvatureDirection direction(3, 1e-12);
  const Eigen::Matrix3d negative_hessian =
    Eigen::Vector3d(0.25, 4.0, 4.0).asDiagonal();

  // Only inverse curvature 1 / 0.25 = 4 clears the cutoff.
  CHECK(direction.refresh(negative_hessian, Eigen::Vector3d::Ones(),
                          10, 2.0));
  CHECK(direction.ready());
  CHECK(direction.rank() == 1);

  const Eigen::VectorXd output = direction.transform(
    Eigen::VectorXd::Ones(1));
  CHECK(near(output(1), 0.0));
  CHECK(near(output(2), 0.0));
  CHECK(near(std::abs(output(0)), std::sqrt(3.0), 1e-7));
}

void test_exact_curvature_uses_metric_scale_and_rank_cap() {
  klhr::CurvatureDirection direction(3, 1e-12);
  const Eigen::Matrix3d negative_hessian =
    Eigen::Vector3d(1.0, 0.25, 0.2).asDiagonal();
  const Eigen::Vector3d metric_scale(2.0, 1.0, 1.0);

  // S H S has spectrum (4, .25, .2). Both broad modes clear the cutoff,
  // but the cap keeps only the third coordinate's inverse curvature 5.
  CHECK(direction.refresh(negative_hessian, metric_scale, 1, 2.0));
  CHECK(direction.rank() == 1);
  const Eigen::VectorXd output = direction.transform(
    Eigen::VectorXd::Ones(1));
  CHECK(near(output(0), 0.0));
  CHECK(near(output(1), 0.0));
  CHECK(near(std::abs(output(2)), std::sqrt(3.0), 1e-7));
}

void test_exact_curvature_dropouts() {
  klhr::CurvatureDirection direction(4, 1e-12);
  CHECK(!direction.refresh(Eigen::Matrix4d::Identity(),
                           Eigen::Vector4d::Ones(), 3, 2.0));
  CHECK(!direction.ready());
  CHECK(direction.dropouts() == 1);

  Eigen::Matrix4d indefinite = Eigen::Matrix4d::Identity();
  indefinite(0, 0) = -0.25;
  CHECK(!direction.refresh(indefinite, Eigen::Vector4d::Ones(), 3, 0.5));
  CHECK(!direction.ready());
  CHECK(direction.dropouts() == 2);
}

} // namespace

int main() {
  test_internal_centering_and_cutoff();
  test_position_trace_controlled_draw();
  test_common_rank_cap();
  test_exact_curvature_finds_broad_mode();
  test_exact_curvature_uses_metric_scale_and_rank_cap();
  test_exact_curvature_dropouts();
  return failures == 0 ? 0 : 1;
}
