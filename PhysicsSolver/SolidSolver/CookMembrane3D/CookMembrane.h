//// @date 2025-01-04
/// @file CookMembrane.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2025 Ma Pengfei
///
/// @brief Cook's membrane
///
///

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawCookMembrane.h"

namespace dolfin {

class CookMembraneSolver
    : public GenericSolidSolver<ConstitutiveLawCookMembrane::FunctionSpace, ConstitutiveLawCookMembrane::BilinearForm,
                                ConstitutiveLawCookMembrane::LinearForm> {
  public:
    std::shared_ptr<SourceScalar> pressure;

    double _kappa;
    double _beta;
    double _G_T;
    double _G_L;
    double _E_L;

    CookMembraneSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> material_types,
                          std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                          double kappa, double beta, double G_T, double G_L, double E_L, std::string result_path)
        : GenericSolidSolver(mesh, result_path), pressure(std::make_shared<SourceScalar>()), _kappa(kappa),
          _beta(beta), _G_T(G_T), _G_L(G_L), _E_L(E_L) {
        LOG_F(INFO, " Initialize CookMembraneSolver.");

        _boundaries = boundaries;
        if (!material_types){
            _material_types = material_types;
        }

        L->pressure = pressure;
        L->A     = std::make_shared<Constant>(1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0));
        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);
        L->G_T   = std::make_shared<Constant>(_G_T);
        L->G_L   = std::make_shared<Constant>(_G_L);
        L->E_L   = std::make_shared<Constant>(_E_L);
    }
    double calculate_q_t(double t) { 
        double T1 = 14;
        if (t < T1) {
            return (-2*(t/T1)*(t/T1)*(t/T1) + 3*(t/T1)*(t/T1))*6.25;
        } else {
            return 1.0*6.25;
        }
 }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {

        // update the traction force
        pressure->data = calculate_q_t(_t);
        printf("pressure->data = %f\n", pressure->data);

        // Define displacement
        // TODO : 将这里的两个变量定义成类的成员，而不是每次调用重新分配内存
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Mark the boundaries
        L->ds = _boundaries;
        if (!_material_types) {
            L->dx = _material_types;
        }

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X) { return -1e20; }
};
} // namespace dolfin
