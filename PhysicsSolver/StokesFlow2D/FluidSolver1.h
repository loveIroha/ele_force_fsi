
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
#include "GenericFluid.h"

namespace FluidSolver1 {

class StaticFlow : public GenericFluid {
  public:
    StaticFlow(BackgroundMesh2D<2> mesh, std::string output_file) : GenericFluid(mesh, output_file) {}

    virtual void reset_bcs() {
        // 设置边界类型
        int ubt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int vbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int pbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};

        double velocity_u[4] = {0, 0, 0, 0};
        double velocity_v[4] = {0, 0, 0, 0};
        double pressure[4]   = {0, 0, 0, 0};

        stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);
    }
};

} // namespace FluidSolver1
