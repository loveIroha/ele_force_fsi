/**
 * @file test_newton_solver.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief here is a demo for the usage of NewtonSolver.
 *        the numerical example is on the page 68 of book "PETSc for Partial
 * differential equations".
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <io/loguru.hpp>
#include <loguru/IBTimer.h>

#include <cmath>

#include "SolidSolver/SolidSolver.h"

class MyProblem : public NonlinearProblem {
  private:
  public:
    virtual void Residual(const std::vector<double>& x, std::vector<double>& r) {
        // IBTimer timer("function residual in class MyProblem");

        CHECK_F(x.size() == r.size(), "Wrong size.");
        r[0] = exp(2.0 * x[0]) / 2.0 - x[1];
        r[1] = x[0] * x[0] + x[1] * x[1] - 1;
    }
};

int main() {
    IBTimer timer("main");

    auto         bicgstab = std::make_shared<BiCGSTAB<std::vector<double>>>(2);
    NewtonSolver ns(bicgstab);
    // auto method = ns.method();
    // LOG_F(INFO, "using %s. ", method.c_str());
    auto my_problem = std::make_shared<MyProblem>();

    //
    std::vector<double> x0(2);
    std::vector<double> b(2);
    auto                nonlinear_result = ns.Solve(my_problem, x0, b);

    LOG_F(WARNING, "Noninear solver successful? %d", nonlinear_result.first);
    LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);

    return 0;
}