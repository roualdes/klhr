#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace klhr {

class OnlinePCA {
public:
  OnlinePCA(std::size_t D = 0,
            std::size_t K = 1,
            double l = 0.0,
            double tol = 1e-10) :
    D_(D),
    K_(K),
    l_(l),
    tol_(tol),
    v_(Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(D),
                             static_cast<Eigen::Index>(K))),
    n_(0) {}

  void update(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
    if (static_cast<std::size_t>(u_in.size()) != D_) {
      throw std::invalid_argument("OnlinePCA::update: input dimension mismatch");
    }

    ++n_;
    Eigen::VectorXd u = u_in;
    const std::size_t ncols = std::min(K_, n_);

    for (std::size_t i = 0; i < ncols; ++i) {
      const Eigen::Index col = static_cast<Eigen::Index>(i);
      if (i == n_ - 1) {
        v_.col(col) = u;
      } else {
        const double w = (static_cast<double>(n_ - 1) - l_) / static_cast<double>(n_);
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
    for (std::size_t i = 0; i < K_; ++i) {
      nv(static_cast<Eigen::Index>(i)) = v_.col(static_cast<Eigen::Index>(i)).norm();
    }

    if (!nv.allFinite()) {
      nv.setZero();
    }

    const double n = static_cast<double>(n_);
    return (n / (n + 5.0)) * nv.array()
      + 1e-3 * (5.0 / (n + 5.0));
  }

  Eigen::MatrixXd vectors() const {
    Eigen::MatrixXd out = v_;
    const Eigen::VectorXd vals = values();
    for (std::size_t i = 0; i < K_; ++i) {
      out.col(static_cast<Eigen::Index>(i)) /= vals(static_cast<Eigen::Index>(i));
    }
    return out;
  }

  void reset() {
    n_ = 0;
    v_.setZero();
  }

  std::size_t count() const {
    return n_;
  }

private:
  std::size_t D_;
  std::size_t K_;
  double l_;
  double tol_;
  Eigen::MatrixXd v_;
  std::size_t n_;
};

} // namespace klhr
