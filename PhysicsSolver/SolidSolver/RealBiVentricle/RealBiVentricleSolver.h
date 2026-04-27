/**
 * @file RealBiVentricleSolver.h
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
double current_pressure_left = 0.0;
double current_pressure_right = 0.0;

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
    WallPressure(double _final_pressure): final_pressure(_final_pressure) {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0]        = final_pressure / 1.0 * std::min(1.0, t);
    }
    // Current time
    double t              = 0.0;
    double final_pressure;
};

class RealBiVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm> 
{
  public:
    std::shared_ptr<WallPressure>  pressure_left;
    std::shared_ptr<WallPressure>  pressure_right;
    double                         _kappa;
    double                         _beta;
    double                         _t = 0.0;

    RealBiVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                             std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                             std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                             std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                             double kappa, double beta, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), 
        pressure_left(std::make_shared<WallPressure>(8*ISUnits::mmHg)),
        pressure_right(std::make_shared<WallPressure>(2*ISUnits::mmHg)),
          _kappa(kappa), _beta(beta) {    
        LOG_F(INFO, " Initialize RealBiVentricleSolver is called!");
        _boundaries = boundaries;
        L->pressure_left = pressure_left;
        L->pressure_right = pressure_right;

        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure_left->t = this->_t;
        pressure_right->t = this->_t;
        std::cout << "pressure_left: " << pressure_left->t*pressure_left->final_pressure << std::endl;
        std::cout << "pressure_right: " << pressure_right->t*pressure_right->final_pressure << std::endl;

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
