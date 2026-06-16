#ifndef MPC_CONTROLLER_HELPERS_H
#define MPC_CONTROLLER_HELPERS_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>

inline double polyeval(const Eigen::VectorXd& coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); ++i) {
        result += coeffs[i] * std::pow(x, i);
    }
    return result;
}

inline Eigen::VectorXd polyfit(const Eigen::VectorXd& xvals,
                               const Eigen::VectorXd& yvals,
                               int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);
    for (int i = 0; i < xvals.size(); ++i) {
        A(i, 0) = 1.0;
    }
    for (int j = 1; j <= order; ++j) {
        for (int i = 0; i < xvals.size(); ++i) {
            A(i, j) = A(i, j-1) * xvals(i);
        }
    }
    auto Q = A.householderQr();
    return Q.solve(yvals);
}

inline std::pair<double, double> globalToLocal(double global_x, double global_y,
                                               double vehicle_x, double vehicle_y,
                                               double vehicle_psi) {
    double dx = global_x - vehicle_x;
    double dy = global_y - vehicle_y;
    double local_x =  dx * std::cos(vehicle_psi) + dy * std::sin(vehicle_psi);
    double local_y = -dx * std::sin(vehicle_psi) + dy * std::cos(vehicle_psi);
    return {local_x, local_y};
}

#endif
