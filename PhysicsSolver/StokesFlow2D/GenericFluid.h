/// @date 2023-12-07
/// @file GenericFluid.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#pragma once
#include <MeshTools/BackgroundMesh2D.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <config.h>
#include <io.h>

class GenericFluid {
  public:
    StokesFlow<2> stokes_flow;

    double  _t = 0.0;
    double& _dt;

    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> v;
    std::vector<std::vector<double>> p;
    std::vector<std::vector<double>> un;
    std::vector<std::vector<double>> vn;
    std::vector<std::vector<double>> uf;
    std::vector<std::vector<double>> vf;

    VTIWriter writer;

    GenericFluid(BackgroundMesh2D<2> mesh, std::string output_file = "fluid/data.pvd")
        : stokes_flow{mesh._Nt, {mesh._Nx, mesh._Ny}, {mesh._Lx, mesh._Ly}, mesh._T, mesh._rho, mesh._mu},
          _dt(stokes_flow._dt), writer{output_file}

    {
        // 设置边界类型
        int ubt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int vbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int pbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};

        double velocity_u[4] = {0, 1, 0, 0};
        double velocity_v[4] = {0, 0, 0, 0};
        double pressure[4]   = {0, 0, 0, 0};

        stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);

        int _Nx = stokes_flow._Nx;
        int _Ny = stokes_flow._Ny;

        u = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        v = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        p = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        un = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vn = make_vector_2D<double>(_Nx + 2, _Ny + 1);

        uf = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vf = make_vector_2D<double>(_Nx + 2, _Ny + 1);
    }

    virtual void reset_bcs() { CHECK_F(false, "'reset_bcs' hasn't been implemented."); }

    auto get_variables() {
        return std::make_tuple(std::ref(u), std::ref(v), std::ref(p), std::ref(un), std::ref(vn), std::ref(uf),
                               std::ref(vf));
    }

    auto get_const_variables() const {
        return std::make_tuple(std::cref(u), std::ref(v), std::ref(p), std::ref(un), std::ref(vn), std::ref(uf),
                               std::ref(vf));
    }
};
