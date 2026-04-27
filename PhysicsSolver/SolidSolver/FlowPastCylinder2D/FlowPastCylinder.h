/// @date 2023-08-03
/// @file FlowPastCylinder.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __FLOW_PAST_CYLINDER_2D_H__
#define __FLOW_PAST_CYLINDER_2D_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"

namespace dolfin {
class FlowPastCylinder : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                                   ConstituitiveLaw::LinearForm> {
  public:
    size_t marker_cylinder = 1;

    double _kappa, _eta, _mu;

    std::shared_ptr<Function> Velocity;

    FlowPastCylinder(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> matrial_types, double kappa,
                     double eta, double mu, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), _kappa(kappa), _eta(eta), _mu(mu) {
        _material_types = matrial_types;

        // 速度
        Velocity = std::make_shared<Function>(V);

        LOG_F(INFO, " FlowPastCylinder is called!");
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // 定义惩罚项函数和向量
        auto                F_penalty = std::make_shared<Function>(V);
        std::vector<double> vector_F_penalty(F_penalty->vector()->local_size());

        L->mu = std::make_shared<Constant>(_mu);
        L->dx = _material_types;

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "cg", "amg");
        // solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);

        // 惩罚位移和速度
        constraint_body(F_penalty, X, Velocity, _kappa, _eta, marker_cylinder);
        F_penalty->vector()->get_local(vector_F_penalty);
        for (size_t i = 0; i < vector_F_penalty.size(); i++) {
            vector_G[i] = vector_G[i] + vector_F_penalty[i];
        }
    }
};
} // namespace dolfin
#endif