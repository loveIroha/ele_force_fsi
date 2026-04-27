/**
 * @file IdealLeftVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-23
 * @date 2023-05-08
 * @date 2023-11-01
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __Ideal_Left_Ventricle_Solver_H__
#define __Ideal_Left_Ventricle_Solver_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawLV.h"

namespace dolfin {

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = final_pressure / 1.0 * std::min(1.0, t);
    }
    // Current time
    double t              = 0.0;
    double final_pressure = 100000.0;
};

class IdealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm>

{
  public:
    std::shared_ptr<WallPressure> pressure;
    double                        _t = 0.0;
    double                        _kappa;
    double                        _beta;
    
    IdealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries,double kappa, double beta,
                             std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), pressure(std::make_shared<WallPressure>()),_kappa(kappa), _beta(beta) {
        LOG_F(INFO, " Initialize IdealLeftVentricleSolver.");
        _boundaries = boundaries;
        L->pressure = pressure;
        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure->t = this->_t;
        std::cout << "pressure: " << pressure->t << std::endl;

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

        // // Solve Ax=b with lumping matrix fastly
        // for (size_t i = 0; i < b.size(); i++)
        //     G_vector[i] = b[i] / mass[i];

        // solve(A, *G->vector(), b, "bicgstab", "amg");
        // for (size_t i = 0; i < b.size(); i++)
        //     G_vector[i] = (*G->vector())[i];
    }
};
} // namespace dolfin
#endif
