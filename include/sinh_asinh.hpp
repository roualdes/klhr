#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <cmath>
#include <algorithm>

struct SASFunctor {
    using Scalar = double;

    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;

    double A_obs;
    double B_obs;

    SASFunctor(double A, double B) : A_obs(A), B_obs(B) {}

    int inputs() const { return 2; }
    int values() const { return 2; }

    static double log_cosh(double x) {
        const double ax = std::abs(x);
        return ax + std::log1p(std::exp(-2.0 * ax)) - std::log(2.0);
    }

    int operator()(const Eigen::VectorXd& theta, Eigen::VectorXd& fvec) const {
        const double alpha = theta[0];
        const double e     = theta[1];

        const double u = std::exp(alpha);
        const double a = std::asinh(u);

        const double A_mod =
            log_cosh(a - e)
            - log_cosh(a + e)
            + 0.5 * std::sinh(2.0 * a) * std::sinh(2.0 * e);

        const double B_mod =
            std::log(1.0 + 2.0 * u * u + std::cosh(2.0 * e))
            - std::log(1.0 + std::cosh(2.0 * e))
            - std::log(1.0 + u * u)
            - u * u * std::cosh(2.0 * e);

        fvec[0] = A_mod - A_obs;
        fvec[1] = B_mod - B_obs;

        return 0;
    }
};
