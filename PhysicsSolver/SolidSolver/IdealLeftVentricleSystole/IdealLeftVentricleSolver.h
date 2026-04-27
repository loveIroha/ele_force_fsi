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
double current_pressure = 0.0;
double current_Tactive  = 0.0;
class FiberDirections : public Expression {
  public:
    // Create expression with 3 components
    FiberDirections(std::shared_ptr<MeshFunction<double>> _c0, std::shared_ptr<MeshFunction<double>> _c1,
                    std::shared_ptr<MeshFunction<double>> _c2)
        : Expression(3), c0(_c0), c1(_c1), c2(_c2) {}

    // Function for evaluating expression on each cell
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        const uint cell_index = cell.index;
        values[0]             = (*c0)[cell_index];
        values[1]             = (*c1)[cell_index];
        values[2]             = (*c2)[cell_index];
    }

    // The data stored in mesh functions
    std::shared_ptr<dolfin::MeshFunction<double>> c0;
    std::shared_ptr<dolfin::MeshFunction<double>> c1;
    std::shared_ptr<dolfin::MeshFunction<double>> c2;
};

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
    double final_pressure = 150000.0;
};

class TensionActive : public dolfin::Expression {
  public:
    // Constructor
    TensionActive() {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0]       = final_tension / 1.0 * std::min(1.0, t);
        current_Tactive = values[0];
    }
    // Current time
    double t             = 0.0;
    double final_tension = 600000.0;
};

class IdealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm>

{
  public:
    std::shared_ptr<WallPressure>  pressure;
    std::shared_ptr<TensionActive> Tactive;
    double                         _kappa;
    double                         _beta;
    double                         _t = 0.0;

    IdealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                             std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                             std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                             std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                             std::shared_ptr<MeshFunction<double>> n00, std::shared_ptr<MeshFunction<double>> n01,
                             std::shared_ptr<MeshFunction<double>> n02, double kappa, double beta,
                             std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), pressure(std::make_shared<WallPressure>()),
          Tactive(std::make_shared<TensionActive>()), _kappa(kappa), _beta(beta) {
        LOG_F(INFO, " Initialize IdealLeftVentricleSolver for systole.");
        _boundaries = boundaries;
        L->pressure = pressure;
        L->Tactive  = Tactive;
        L->kappa    = std::make_shared<Constant>(_kappa);
        L->beta     = std::make_shared<Constant>(_beta);

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
        L->n0 = std::make_shared<FiberDirections>(n00, n01, n02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure->t = this->_t;
        Tactive->t  = this->_t;
        printf("current time: %.5f, current pressure: %.5e, current Tactive: %.5e\n", this->_t, current_pressure,
               current_Tactive);
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
