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

#include "ConstitutiveLawLVMV.h"

namespace dolfin {

// 是一个函数
double current_pressure(double t, double p_load, double t_load, double t_end_diastole) {
    double p = 0.0;
    if (t < t_load) {
        p = p_load * t / t_load;
    } else if (t < t_end_diastole) {
        p = p_load * (t_end_diastole- t) / (t_end_diastole - t_load);
    }
    return p * ISUnits::mmHg;
}

class RealMitralValveSolver
    : public GenericSolidSolver<ConstitutiveLawLVMV::FunctionSpace, ConstitutiveLawLVMV::BilinearForm,
                                ConstitutiveLawLVMV::LinearForm> {
  public:
    double _kappa;
    double _beta;
    double _t_end_diastole;
    double _p_load = 8.0;
    double _t_load = 0.4;

    std::shared_ptr<Source> _t_current = nullptr;
    std::shared_ptr<Source> _pressure  = nullptr;

    RealMitralValveSolver(std::shared_ptr<Mesh> mesh, 
                          std::shared_ptr<MeshFunction<std::size_t>> material_types,
                          std::shared_ptr<MeshFunction<std::size_t>> boundaries, 
                          std::shared_ptr<FiberDirections> f0,
                          std::shared_ptr<FiberDirections> s0, 
                          double kappa, double beta, double t_end_diastole,
                          std::string result_path)
        : GenericSolidSolver(mesh, result_path), _kappa(kappa), _beta(beta), _t_end_diastole(t_end_diastole) {

        LOG_F(INFO, "RealMitralValveSolver is called!");

        // Initialize
        _material_types = material_types;
        _boundaries     = boundaries;
        _t_current      = std::make_shared<Source>();
        _pressure       = std::make_shared<Source>();

        // Mark the boundaries
        L->ds = _boundaries;
        L->dx = _material_types;

        // Set the coefficients
        // L->kappa           = std::make_shared<Constant>(_kappa);
        // L->beta            = std::make_shared<Constant>(_beta);
        // L->t_current       = _t_current;
        L->pressure = _pressure;
        // L->t_end_diastole = std::make_shared<Constant>(_t_end_diastole);

        // Define fiber directions
        // L->f0 = f0;
        // L->s0 = s0;
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {

        // Define displacement
        // TODO : 将这里的两个变量定义成类的成员，而不是每次调用重新分配内存
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        _pressure->data = current_pressure(_t, _p_load, _t_load, _t_end_diastole);
        
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
