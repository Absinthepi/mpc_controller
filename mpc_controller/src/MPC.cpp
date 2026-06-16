#include "mpc_controller/MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>
#include <vector>

using CppAD::AD;

size_t N = 10;
double dt = 0.1;

const double Lf = 2.67;
double ref_v = 40.0;

const double w_cte   = 2000.0;
const double w_epsi  = 2000.0;
const double w_v     = 1.0;
const double w_delta = 5.0;
const double w_a     = 5.0;
const double w_ddelta = 200.0;
const double w_da     = 10.0;

const size_t x_start   = 0;
const size_t y_start   = N;
const size_t psi_start = 2 * N;
const size_t v_start   = 3 * N;
const size_t cte_start = 4 * N;
const size_t epsi_start = 5 * N;
const size_t delta_start = 6 * N;
const size_t a_start     = 6 * N + (N - 1);

class FG_eval {
public:
    Eigen::VectorXd coeffs;
    FG_eval(Eigen::VectorXd coeffs) : coeffs(coeffs) {}

    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    void operator()(ADvector& fg, const ADvector& vars) {
        fg[0] = 0;
        for (size_t i = 0; i < N; ++i) {
            fg[0] += w_cte   * CppAD::pow(vars[cte_start + i], 2);
            fg[0] += w_epsi  * CppAD::pow(vars[epsi_start + i], 2);
            fg[0] += w_v     * CppAD::pow(vars[v_start + i] - ref_v, 2);
        }

        for (size_t i = 0; i < N - 1; ++i) {
            fg[0] += w_delta * CppAD::pow(vars[delta_start + i], 2);
            fg[0] += w_a     * CppAD::pow(vars[a_start + i], 2);
        }

        for (size_t i = 0; i < N - 2; ++i) {
            fg[0] += w_ddelta * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
            fg[0] += w_da     * CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
        }

        fg[1] = vars[x_start];
        fg[2] = vars[y_start];
        fg[3] = vars[psi_start];
        fg[4] = vars[v_start];
        fg[5] = vars[cte_start];
        fg[6] = vars[epsi_start];

        for (size_t i = 0; i < N - 1; ++i) {

            AD<double> x0    = vars[x_start + i];
            AD<double> y0    = vars[y_start + i];
            AD<double> psi0  = vars[psi_start + i];
            AD<double> v0    = vars[v_start + i];
            AD<double> cte0  = vars[cte_start + i];
            AD<double> epsi0 = vars[epsi_start + i];

            AD<double> delta0 = vars[delta_start + i];
            AD<double> a0     = vars[a_start + i];

            // 下一时刻状态
            AD<double> x1    = vars[x_start + i + 1];
            AD<double> y1    = vars[y_start + i + 1];
            AD<double> psi1  = vars[psi_start + i + 1];
            AD<double> v1    = vars[v_start + i + 1];
            AD<double> cte1  = vars[cte_start + i + 1];
            AD<double> epsi1 = vars[epsi_start + i + 1];

            AD<double> f0 = coeffs[0];
            for (int k = 1; k < coeffs.size(); ++k)
                f0 += coeffs[k] * CppAD::pow(x0, k);

            AD<double> f_prime = coeffs[1];
            for (int k = 2; k < coeffs.size(); ++k)
                f_prime += AD<double>(k) * coeffs[k] * CppAD::pow(x0, k - 1);

            AD<double> psides0 = CppAD::atan(f_prime);

            size_t base = 7 + i * 6;
            fg[base + 0] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
            fg[base + 1] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
            fg[base + 2] = psi1 - (psi0 + v0 / Lf * delta0 * dt);
            fg[base + 3] = v1 - (v0 + a0 * dt);
            fg[base + 4] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
            fg[base + 5] = epsi1 - ((psi0 - psides0) + v0 / Lf * delta0 * dt);
        }
    }
};

MPC::MPC() {}
MPC::~MPC() {}

std::vector<double> MPC::Solve(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &coeffs) {
    typedef CPPAD_TESTVECTOR(double) Dvector;

    size_t n_vars = 6 * N + 2 * (N - 1);
    size_t n_constraints = 6 + 6 * (N - 1);

    
    Dvector vars(n_vars);
    for (size_t i = 0; i < n_vars; ++i) vars[i] = 0.0;

    vars[x_start]    = state[0];
    vars[y_start]    = state[1];
    vars[psi_start]  = state[2];
    vars[v_start]    = state[3];
    vars[cte_start]  = state[4];
    vars[epsi_start] = state[5];

    
    Dvector vars_lowerbound(n_vars), vars_upperbound(n_vars);
    for (size_t i = 0; i < n_vars; ++i) {
        vars_lowerbound[i] = -1.0e19;
        vars_upperbound[i] =  1.0e19;
    }

    
    for (size_t i = 0; i < N - 1; ++i) {
        vars_lowerbound[delta_start + i] = -0.4363;  // ±25°
        vars_upperbound[delta_start + i] =  0.4363;
        vars_lowerbound[a_start + i] = -1.0;         // 加速度 ±1 m/s²
        vars_upperbound[a_start + i] =  1.0;
    }

    Dvector constraints_lowerbound(n_constraints), constraints_upperbound(n_constraints);
    for (size_t i = 0; i < n_constraints; ++i) {
        constraints_lowerbound[i] = 0.0;
        constraints_upperbound[i] = 0.0;
    }

    FG_eval fg_eval(coeffs);

    std::string options;
    options += "Integer print_level  0\n";
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    options += "Numeric max_cpu_time          0.5\n";

    CppAD::ipopt::solve_result<Dvector> solution;
    CppAD::ipopt::solve<Dvector, FG_eval>(
        options, vars, vars_lowerbound, vars_upperbound,
        constraints_lowerbound, constraints_upperbound,
        fg_eval, solution);

    if (solution.status != CppAD::ipopt::solve_result<Dvector>::success) {
        std::cerr << "[MPC] Solver failed, status = " << solution.status << std::endl;
        return {};  
    }


    return {solution.x[delta_start], solution.x[a_start]};
}