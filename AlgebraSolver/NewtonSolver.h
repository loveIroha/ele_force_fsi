/**
 * @file NewtonSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-09
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

// error control : $ \frac{\|g(x^{(k)})\|_2}{\|x^{(k)}\|_2} $
// when \|x^{(k)}\|_2 = 0, set \|x^{(k)}\|_2 = 1
// where \|x^{(k)}\|_2=\sqrt{x^{(k)}\cdot x^{(k)}}

#ifndef _NEWTON_SOLVER_
#define _NEWTON_SOLVER_

#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "NewtonSolver.h"
#include "NonlinearProblem.h"
#include "NonlinearSolver.h"

template <typename VectorType>
class NewtonSolver : public NonlinearSolver<VectorType> {
  public:
    NewtonSolver(std::shared_ptr<LinearSolver<VectorType>> _linear_solver)
        : NonlinearSolver<VectorType>::NonlinearSolver(_linear_solver) {}

    virtual std::string method() const final { return "Default newton solver"; };

    // TODO : param b is useless here.
    virtual std::pair<bool, std::pair<double, int>>
    Solve(std::shared_ptr<NonlinearProblem<VectorType>> nonlinear_problem, std::shared_ptr<VectorType> x0,
          std::shared_ptr<const VectorType> b) final {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Solve nonlinear problem with Newton method");

        auto linear_solver = this->get_linear_solver();
        CHECK_F(linear_solver->problem_size() == x0->size(), "Wrong size!");

        // NOTE : It will allocate memory for nonlinear_problem. For efficiency, it
        // will not reallocate memory when VectorType = GpuVector if vectors exist.
        nonlinear_problem->resize(x0->size());

        auto xk       = nonlinear_problem->get_xk();
        auto rhs      = nonlinear_problem->get_rhs();
        auto delta_x  = nonlinear_problem->get_delta_x();
        auto residual = nonlinear_problem->get_residual();

        double epsilon     = std::sqrt(std::numeric_limits<double>::epsilon()) * 10.0;
        double norm_r      = epsilon;
        double norm_x      = epsilon;
        int    linear_iter = 0;
        int    newton_iter = 0;

        // NOTE : Set initial value for newton iteration.
        *xk = *x0;
        try {
            nonlinear_problem->Residual(rhs);
        } catch (std::exception& e) {
            LOG_F(ERROR, "Residual error");
            return std::make_pair(false, std::make_pair(10e10, 10e10));
        }

        norm_r = std::sqrt(rhs->inner(*rhs));
        norm_x = std::max(std::sqrt(xk->inner(*xk)), epsilon);
        LOG_F(WARNING, "at the start, the residual is %.12e, the relative residual is %.12e.", norm_r, norm_r / norm_x);
        // if (norm_r/norm_x < this->max_nonlinear_tolerance) return
        // std::make_pair(true, std::make_pair(norm_r/norm_x,linear_iter));
        do {
            LOG_SCOPE_F(INFO, "Inside the NEWTON iteration, iter = %d, norm_r = %e, norm_x = %e.", newton_iter, norm_r,
                        norm_x);
            // NOTE : rhs = h(xk),so we solve A(-dx)=rhs, and finnally xk = xk -
            // (-dx); NOTE : initial guess for the linear solver.
            *delta_x = *rhs;

            // solve delta_x, which is (-dx) actually.
            std::pair<bool, std::pair<double, int>> linear_result;
            try {
                linear_result = linear_solver->Solve(nonlinear_problem, delta_x, rhs);
            } catch (std::exception& e) {
                LOG_F(ERROR, "Linear solver error");
                return std::make_pair(false, std::make_pair(10e10, 10e10));
            }

            newton_iter++;
            if (linear_result.first)
                LOG_F(WARNING, "Linear solver succeed.");
            else {
                LOG_F(WARNING, "Linear solver failed.");
                break;
            }
            LOG_F(WARNING, "Linear solver residual : %.12e, linear_iter : %d", linear_result.second.first,
                  linear_result.second.second);
            linear_iter += linear_result.second.second;

            // update xk = xk - (-dx) and calculate the residual.
            xk->axpy(-1.0, *delta_x, *xk);

            try {
                nonlinear_problem->Residual(rhs);
            } catch (std::exception& e) {
                LOG_F(ERROR, "Residual error");
                return std::make_pair(false, std::make_pair(10e10, 10e10));
            }

            norm_r = std::sqrt(rhs->inner(*rhs));
            norm_x = std::max(std::sqrt(xk->inner(*xk)), epsilon);

            LOG_F(WARNING, "after %d iteration, the residual is %.12e, relative residual is %.12e.", linear_iter,
                  norm_r, norm_r / norm_x);

            // update the result as return.
            *x0 = *xk;

            if (norm_r / norm_x < this->max_nonlinear_tolerance)
                return std::make_pair(true, std::make_pair(norm_r / norm_x, linear_iter));
        } while (linear_iter < this->max_nonlinear_iteration);

        return std::make_pair(false, std::make_pair(std::sqrt(norm_r / norm_x), linear_iter));
    };

    virtual ~NewtonSolver() {}
};

#endif

// double linesearch(
//     std::shared_ptr<NonlinearProblem> nonlinear_problem,
//     std::vector<double> &x0,
//     const std::vector<double> &delta_x,
//     double norm_start
// ){
//     std::vector<double> x = x0;
//     std::vector<double> rhs = x0;
//     double alpha = 1.0;
//     double norm_r = 0.0;
//     for (size_t i = 1; i < 4; i++)
//     {
//         alpha = 1 - std::pow(0.5, i);
//         linear_solver->axpy(x, delta_x, x0, alpha);
//         nonlinear_problem->Residual(x, rhs);
//         norm_r = sqrt(linear_solver->dot(rhs, rhs));
//         LOG_F(WARNING, "In Newton line search, alpha is %.12e, residual is
//         %.12e.", alpha, norm_r); if(norm_r < norm_start) break;
//     }
//     LOG_F(WARNING, "Returned norm %.12e.", norm_r);
//     return norm_r;
// }

// // linesearch.
// if (norm_r < norm_r_temp) {
//     // change xk.
//     norm_r = linesearch(nonlinear_problem, xk, delta_x, norm_r);
//     LOG_F(WARNING, "Returned norm(2) %.12e.", norm_r);
// }
// else {
//     norm_r = norm_r_temp;
// }