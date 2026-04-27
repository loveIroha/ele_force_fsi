//// @date 2024-04-20
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

class RealMitralValveSolver
    : public GenericSolidSolver<ConstitutiveLawMV::FunctionSpace, ConstitutiveLawMV::BilinearForm,
                                ConstitutiveLawMV::LinearForm> {
  public:
    double _kappa;
    double _beta;
    double _t_end_diastole;

    double _t_start_closing;
    double _dilation_radius;

    double _posterior_papillary_x;
    double _posterior_papillary_y;
    double _posterior_papillary_z;

    double _anterior_papillary_x;
    double _anterior_papillary_y;
    double _anterior_papillary_z;

    std::shared_ptr<Source> _t_current = nullptr;

    RealMitralValveSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> material_types,
                          std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                          std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                          std::shared_ptr<MeshFunction<double>> f02, double kappa, double beta, double t_start_closing,
                          double dilation_radius, double posterior_papillary_x, double posterior_papillary_y,
                          double posterior_papillary_z, double anterior_papillary_x, double anterior_papillary_y,
                          double anterior_papillary_z, std::string result_path)
        : GenericSolidSolver(mesh, result_path), _kappa(kappa), _beta(beta), _t_start_closing(t_start_closing),
          _dilation_radius(dilation_radius), _posterior_papillary_x(posterior_papillary_x),
          _posterior_papillary_y(posterior_papillary_y), _posterior_papillary_z(posterior_papillary_z),
          _anterior_papillary_x(anterior_papillary_x), _anterior_papillary_y(anterior_papillary_y),
          _anterior_papillary_z(anterior_papillary_z) {

        LOG_F(INFO, "RealMitralValveSolver is called!");

        // Initialize
        _material_types = material_types;
        _boundaries     = boundaries;
        _t_current      = std::make_shared<Source>();

        // Mark the boundaries
        L->ds = _boundaries;
        L->dx = _material_types;

        // Set the coefficients
        L->kappa           = std::make_shared<Constant>(_kappa);
        L->beta            = std::make_shared<Constant>(_beta);
        L->t_current       = _t_current;
        L->t_start_closing = std::make_shared<Constant>(_t_start_closing);

        L->posterior_papillary_x = std::make_shared<Constant>(_posterior_papillary_x);
        L->posterior_papillary_y = std::make_shared<Constant>(_posterior_papillary_y);
        L->posterior_papillary_z = std::make_shared<Constant>(_posterior_papillary_z);

        L->anterior_papillary_x = std::make_shared<Constant>(_anterior_papillary_x);
        L->anterior_papillary_y = std::make_shared<Constant>(_anterior_papillary_y);
        L->anterior_papillary_z = std::make_shared<Constant>(_anterior_papillary_z);
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
