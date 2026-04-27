/**
 * @file RealLeftVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-28
 * @date 2023-11-02 适应新的流体求解器
 * @date 2024-03-12 适应新的IBFE系统
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawLV.h"

namespace dolfin {
double current_pressure = 0.0;
// TODO: 接受一个参数
class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0]        = final_pressure / 1.0 * std::min(1.0, t);
        current_pressure = values[0];
    }
    // Current time
    double t              = 0.0;
    double final_pressure = 10665.789; // 8mmHg in dyn/cm²
};
class RealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm> {
  public:
    double _kappa;
    double _beta;

    std::shared_ptr<WallPressure> pressure;

    RealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> material_types,
                            std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                            std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                            std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                            std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                            double kappa, double beta, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), pressure(std::make_shared<WallPressure>()), 
          _kappa(kappa), _beta(beta) {
        
        LOG_F(INFO, "RealLeftVentricleSolver is called!");

        // Initialize
        _material_types = material_types;
        _boundaries     = boundaries;
        L->pressure = pressure;

        // Mark the boundaries
        L->ds = _boundaries;
        // L->dx = _material_types;

        // Set the coefficients
        L->kappa           = std::make_shared<Constant>(_kappa);
        L->beta            = std::make_shared<Constant>(_beta);

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure->t = this->_t;

        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X) { return -1e20; }
};
} // namespace dolfin
