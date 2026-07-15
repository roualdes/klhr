#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>

namespace klhr {

class OnlinePCA {
public:
  OnlinePCA(Eigen::Index D = 0,
            Eigen::Index K = 1,
            double l = 0.0,
            double tol = 1e-10) :
    D_(checked_dimension_(D)),
    K_(checked_dimension_(K)),
    l_(l),
    tol_(tol),
    v_(Eigen::MatrixXd::Zero(D_, K_)),
    n_(0) {}

  void update(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
    if (u_in.size() != D_) {
      throw std::invalid_argument("OnlinePCA::update: input dimension mismatch");
    }

    ++n_;
    Eigen::VectorXd u = u_in;
    const Eigen::Index ncols = std::min(K_, n_);

    for (Eigen::Index col = 0; col < ncols; ++col) {
      if (col == n_ - 1) {
        v_.col(col) = u;
      } else {
        const double w = (n_ - 1 - l_) / n_;
        Eigen::VectorXd v = v_.col(col);
        double nv = v.norm();
        v_.col(col) = w * v + (1.0 - w) * u * (u.dot(v) / (nv + tol_));

        v = v_.col(col);
        nv = v.norm();
        u -= u.dot(v) * v / (nv * nv + tol_);
      }
    }
  }

  Eigen::VectorXd values() const {
    Eigen::VectorXd nv(K_);
    for (Eigen::Index i = 0; i < K_; ++i) {
      nv(i) = v_.col(i).norm();
    }

    if (!nv.allFinite()) {
      nv.setZero();
    }

    return nv; // (n / (n + 5.0)) * nv.array() + 1e-3 * (5.0 / (n + 5.0));
  }

  Eigen::MatrixXd vectors() const {
    Eigen::MatrixXd out = v_;
    const Eigen::VectorXd vals = values();
    for (Eigen::Index i = 0; i < K_; ++i) {
      out.col(i) /= vals(i);
    }
    return out;
  }

  void reset() {
    n_ = 0;
    v_.setZero();
  }

  Eigen::Index count() const {
    return n_;
  }

private:
  static Eigen::Index checked_dimension_(const Eigen::Index value) {
    if (value < 0) {
      throw std::invalid_argument("OnlinePCA: dimensions must be nonnegative");
    }
    return value;
  }

  Eigen::Index D_;
  Eigen::Index K_;
  double l_;
  double tol_;
  Eigen::MatrixXd v_;
  Eigen::Index n_;
};

} // namespace klhr
