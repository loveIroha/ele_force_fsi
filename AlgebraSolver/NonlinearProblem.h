/**
 * @file NonlinearProblem.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-09
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _NONLINEAR_PROBLEM_
#define _NONLINEAR_PROBLEM_

#include <limits>

#include "LinearProblem.h"
#include "LinearSolver.h"
#include "StdVector.h"

template <typename VectorType>
class NonlinearProblem : public LinearProblem<VectorType> {
  private:
    std::shared_ptr<VectorType> xk;
    std::shared_ptr<VectorType> rhs;
    std::shared_ptr<VectorType> delta_x;
    std::shared_ptr<VectorType> residual;

    std::shared_ptr<VectorType> r1;
    std::shared_ptr<VectorType> r2;
    std::shared_ptr<VectorType> xk_plus_epsilon_x;

  public:
    std::shared_ptr<VectorType> get_xk() { return xk; }
    std::shared_ptr<VectorType> get_rhs() { return rhs; }
    std::shared_ptr<VectorType> get_delta_x() { return delta_x; }
    std::shared_ptr<VectorType> get_residual() { return residual; }

    void resize(size_t __size) {
        LOG_SCOPE_FUNCTION(INFO);
        xk                = std::make_shared<VectorType>();
        rhs               = std::make_shared<VectorType>();
        delta_x           = std::make_shared<VectorType>();
        residual          = std::make_shared<VectorType>();
        r1                = std::make_shared<VectorType>();
        r2                = std::make_shared<VectorType>();
        xk_plus_epsilon_x = std::make_shared<VectorType>();

        xk->resize(__size);
        rhs->resize(__size);
        delta_x->resize(__size);
        residual->resize(__size);
        r1->resize(__size);
        r2->resize(__size);
        xk_plus_epsilon_x->resize(__size);
    }
    NonlinearProblem() {}

    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> r) final {
        // $$
        // \boldsymbol{J}(\boldsymbol{X}) \boldsymbol{y} \approx
        // \frac{\mathbf{g}(\boldsymbol{X}+\epsilon
        // \boldsymbol{y})-\mathbf{g}(\boldsymbol{X})}{\epsilon}
        // $$

        // $$
        // \epsilon=\frac{\sqrt{(1+\|\boldsymbol{X}\|) \epsilon_{\text {mach
        // }}}}{\|\boldsymbol{y}\|}
        // $$

        // used to calculate Jx = (h(x+epsilon*x0)-h(x))/epsilon
        // NOTE : epsilon is a sensitive paramters, it should be larger than the
        // square root of machine error.
        double epsilon = std::sqrt(std::numeric_limits<double>::epsilon()) * 10.0;
        epsilon *= (1.0 + std::sqrt(xk->inner(*xk))) / std::sqrt(x->inner(*x));
        xk_plus_epsilon_x->axpy(epsilon, *x, *xk);
        LOG_F(WARNING, "Nonlinear form with epsilon = %.12e.", epsilon);

        // Jx = (h(x+epsilon*x0)-h(x))/epsilon
        // Residual(xk, r1);
        *r1 = *rhs;
        Residual(xk_plus_epsilon_x, r2);

        r->axpy(-1, *r1, *r2);
        r->axpy(1.0 / epsilon - 1, *r);
    }

    virtual void Residual(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> r) = 0;

    void Residual(std::shared_ptr<VectorType> r) { Residual(xk, r); }

    virtual ~NonlinearProblem() {}
};

#endif