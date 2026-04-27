/**
 * @file ConjugateGradient.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-04
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

// Conjugate Gradient is easy to implement, here is the algorithm:
// http://minio.mapengfei.xyz:80/documentation-pictures/pics/2022/5/12/image-20220512165741835_repeat_1652345867796__133835.png

#ifndef _CONJUGATEGRADIENT_H_
#define _CONJUGATEGRADIENT_H_

#include <cmath>

#include "LinearSolver.h"

template <typename VectorType>
class ConjugateGradient : public LinearSolver<VectorType> {
  private:
    std::shared_ptr<VectorType> r, p, temp;

  public:
    ConjugateGradient(size_t n) { Initialize(n); }

    virtual ~ConjugateGradient() {}

    virtual void Initialize(size_t n) final {
        this->_problem_size = n;

        r    = std::make_shared<VectorType>();
        p    = std::make_shared<VectorType>();
        temp = std::make_shared<VectorType>();

        r->resize(n);
        p->resize(n);
        temp->resize(n);
    }

    // 控制误差有三个指标：
    // $$
    // \|\boldsymbol r_k\|,\frac{\|\boldsymbol r_k\|}{\|\boldsymbol
    // r_0\|},\frac{\|\boldsymbol r_k\|}{\|\boldsymbol b\|}
    // $$
    // $\|\boldsymbol r_k\|$ 控制绝对误差，没有什么用。
    // $ \frac{\|\boldsymbol r_k\|}{\|\boldsymbol r_0\|} $
    // 可以控制残差下降的程度。 $ \frac{\|\boldsymbol r_k\|}{\|\boldsymbol b\|}$
    // 可以控制相对误差。

    virtual std::pair<bool, std::pair<double, int>> Solve(std::shared_ptr<LinearProblem<VectorType>> linear_problem,
                                                          std::shared_ptr<VectorType>                x0,
                                                          std::shared_ptr<const VectorType>          b) final override {
        int  iter = 0;
        auto kmax = this->get_max_iteration();
        auto rtol = this->get_tolerance();

        // r = b - A*x0 is divided into two steps:
        linear_problem->form(x0, temp); // temp = Ax
        r->axpy(-1.0, *temp, *b);       // r = b - temp
        *p = *r;

        double rnorm0 = r->inner(*r);
        double rnorm  = rnorm0;

        if (!this->silent) LOG_F(INFO, "Conjugate Gradient solver : tolerance %.12e, maxiterd %d.", rtol, iter);

        if (rnorm < rtol * rtol) return std::make_pair(true, std::make_pair(rnorm, 0));
        while (iter < kmax) {
            ++iter;
            if (!this->silent) LOG_F(INFO, "Conjugate Gradient solver(start) : iteration %d.", iter);
            linear_problem->form(p, temp);          // temp = Ap
            double alpha = rnorm / p->inner(*temp); // alpha = rr / pAp
            x0->axpy(alpha, *p, *x0);
            r->axpy(-alpha, *temp, *r);
            double rnorm_new = r->inner(*r);
            double beta      = rnorm_new / rnorm;
            rnorm            = rnorm_new;
            if (!this->silent)
                LOG_F(INFO, "Conjugate Gradient solver(end)   : iteration %d, relative_error norm : %.12e.", iter,
                      rnorm);
            if (rnorm / rnorm0 < rtol * rtol) return std::make_pair(true, std::make_pair(rnorm, iter));
            p->axpy(beta, *p, *r);
        }
        return std::make_pair(false, std::make_pair(rnorm, iter));
    }
};

#endif