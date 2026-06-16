#ifndef MPC_H
#define MPC_H

#include <vector>
#include <Eigen/Dense>

class MPC {
public:
    MPC();

    virtual ~MPC();

    
    std::vector<double> Solve(const Eigen::VectorXd &state,
                              const Eigen::VectorXd &coeffs);
};

#endif 