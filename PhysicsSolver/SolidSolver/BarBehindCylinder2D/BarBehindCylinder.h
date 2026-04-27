/// @date 2023-09-15
/// @file BarBehindCylinder.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __BAR_BEHIND_CYLINDER_2D_H__
#define __BAR_BEHIND_CYLINDER_2D_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"

namespace dolfin {
class BarBehindCylinder : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                                    ConstituitiveLaw::LinearForm> {
  public:
    size_t marker_disk         = 1;
    size_t marker_beam         = 2;
    size_t marker_disk_surface = 3;

    double _kappa, _eta, _c1_s, _G_s, _nv;

    std::shared_ptr<Function> Velocity;

    BarBehindCylinder(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> matrial_types,
                      std::shared_ptr<MeshFunction<std::size_t>> boundaries, double kappa, double eta, double c1_s,
                      double G_s, double nv, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), _kappa(kappa), _eta(eta), _c1_s(c1_s), _G_s(G_s), _nv(nv) {
        _material_types = matrial_types;
        _boundaries     = boundaries;

        // 速度
        Velocity = std::make_shared<Function>(V);

        LOG_F(INFO, " BarBehindCylinder is called!");
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // find_marked_cell(2);
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        L->c1_s = std::make_shared<Constant>(_c1_s);
        L->G_s  = std::make_shared<Constant>(_G_s);
        L->nv   = std::make_shared<Constant>(_nv);
        L->dx   = _material_types;

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);

        // 惩罚位移和速度
        auto F_penalty = std::make_shared<Function>(V);
        constraint_body(F_penalty, X, Velocity, _kappa, _eta, marker_disk);

        std::vector<double> vector_F_penalty(F_penalty->vector()->local_size());
        F_penalty->vector()->get_local(vector_F_penalty);

        auto F_penalty_surface = std::make_shared<Function>(V);
        constraint_facets(F_penalty_surface, X, Velocity, _kappa, _eta, marker_disk_surface);
        std::vector<double> vector_F_penalty_surface(F_penalty_surface->vector()->local_size());
        F_penalty_surface->vector()->get_local(vector_F_penalty_surface);

        for (size_t i = 0; i < vector_F_penalty.size(); i++) {
            vector_G[i] = vector_G[i] + vector_F_penalty[i] + vector_F_penalty_surface[i];
        }
    }
};
} // namespace dolfin
#endif