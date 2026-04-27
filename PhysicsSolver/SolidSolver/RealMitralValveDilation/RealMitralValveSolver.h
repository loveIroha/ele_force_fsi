/// @date 2024-04-20
/// @file RealMitralValveSolver.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief 管道内的二尖瓣模拟
///
///

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawMV.h"

namespace dolfin {

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

class Source : public dolfin::Expression {
  public:
    double data;
    Source() : data(0.0) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = data;
    }
};

class RealMitralValveSolver
    : public GenericSolidSolver<ConstitutiveLawMV::FunctionSpace, ConstitutiveLawMV::BilinearForm,
                                ConstitutiveLawMV::LinearForm> {
  public:
    double _kappa;
    double _beta;
    double _t_end_diastole;
    double _t_start_closing;
    double _dilation_radius;
    std::shared_ptr<Source> _t_current = nullptr;

    RealMitralValveSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> material_types,
                          std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                          std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                          std::shared_ptr<MeshFunction<double>> f02, 
                          double kappa, double beta, double t_start_closing, double dilation_radius,
                          std::string result_path)
        : GenericSolidSolver(mesh, result_path), _kappa(kappa), _beta(beta), _t_start_closing(t_start_closing),
        _dilation_radius(dilation_radius) {
        LOG_F(INFO, "RealMitralValveSolver is called!");

        // Initialize
        _material_types = material_types;
        _boundaries     = boundaries;
        _t_current      = std::make_shared<Source>();

        // Mark the boundaries
        L->ds = _boundaries;
        L->dx   = _material_types;
        
        // Set the coefficients
        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);
        L->t_current = _t_current;
        L->t_start_closing = std::make_shared<Constant>(_t_start_closing);  
        L->dilation_radius = std::make_shared<Constant>(_dilation_radius); 
        // L->t_end_closing = std::make_shared<Constant>(0.6);   
        // L->t_systole = std::make_shared<Constant>(0.2);   

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {

        // Define displacement
        // TODO : 将这里的两个变量定义成类的成员，而不是每次调用重新分配内存
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        _t_current->data = _t;
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
