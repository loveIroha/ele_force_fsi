/// @date 2023-08-03
/// @file FlowpushCube2D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __FLOW_PAST_CUBE_2D_H__
#define __FLOW_PAST_CUBE_2D_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"

namespace dolfin {
class FlowpushCube2D : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                                 ConstituitiveLaw::LinearForm> {
  public:
    double _eta, _beta, _dt;
    // NOTE: X_nm1 = X_{n-1}, 需要在每个时间步的最后设置
    // NOTE: _beta, _eta 为经验参数
    std::shared_ptr<Function> X_nm1;
    FlowpushCube2D(std::shared_ptr<Mesh> mesh, double eta, double beta, double dt, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), _eta(eta), _beta(beta), _dt(dt) {
        // X_nm1 = std::make_shared<Function>(V);
        // L->Xn = X_nm1;
        LOG_F(INFO, " FlowpushCube2D is called!");
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // L->dt = std::make_shared<Constant>(_dt);
        // L->eta = std::make_shared<Constant>(_eta);
        // L->beta = std::make_shared<Constant>(_beta);
        L->mu = std::make_shared<Constant>(_beta);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);

        // X_{n-1} = X_{n} 赋值位移
        // TODO: 隐式求解时不能这样做
        // *X_nm1->vector() = *X->vector();
    }
};
} // namespace dolfin
#endif