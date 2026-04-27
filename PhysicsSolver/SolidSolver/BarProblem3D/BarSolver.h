/**
 * @file BarSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-18
 *       2023-10-31
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawBar.h"
// #include "EnergyNorm.h"

namespace dolfin {

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = final_pressure / 0.1 * std::min(0.1, t);
    }
    // Current time
    double t              = 0.0;
    double final_pressure = 40.0;
};

class BarSolver : public GenericSolidSolver<ConstitutiveLawBar::FunctionSpace, ConstitutiveLawBar::BilinearForm,
                                            ConstitutiveLawBar::LinearForm> {
  public:
    std::shared_ptr<WallPressure> pressure;
    double                        _t = 0.0;
    double                        _kappa;
    double                        _beta;

    BarSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries, double kappa, double beta,
              std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), pressure(std::make_shared<WallPressure>()),_kappa(kappa), _beta(beta) {
        LOG_F(INFO, " Initialize BarSolver.");

        _boundaries = boundaries;
        L->pressure = pressure;
        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure->t = this->_t;
        std::cout << "pressure: " << pressure->t * 40 << std::endl;

        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Mark the boundaries
        L->ds = _boundaries;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X)
    {
    //     // auto X = std::make_shared<Function>(V);
    //     // X->vector()->set_local(vector_X);
    //     // EnergyNorm::Form_M1 energy_norm(_mesh);
    //     // energy_norm.X = X;
    //     // return assemble(energy_norm);
        return -1e20;
    }
};
} // namespace dolfin
