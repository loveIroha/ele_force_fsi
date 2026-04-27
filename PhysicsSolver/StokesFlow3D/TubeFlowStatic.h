/// @date 2024-04-20
/// @file TubeFlow.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include "Kokkos/StokesFlow3D_kokkos.h"

namespace stokes_flow {
template <int DIM>
class TubeFlow : public StokesFlow<DIM> {
  public:
    std::vector<double> input_pressure;           // 输入的压力数据
    double              dt_input_pressuret;       // 压力数据的时间间隔
    double              t0_input_pressuret = 0.0; // 压力数据的初始时间
    double              radius             = 3.0; // 圆心为(5.0, 5.0, z)，半径为3.0
    double              center_x           = 5.0;
    double              center_y           = 5.0;
    double              rho                = 1.0; // 流体密度

    TubeFlow(std::shared_ptr<NavierStokesDemo> _ns_demo, std::string _output_file)
        : StokesFlow<DIM>(_ns_demo, _output_file + "fluid/data.pvd") {
        // 设置边界类型和边界值
        std::array<int, 6> all_boundary_type   = {1, 1, 1, 1, 2, 2};
        std::array<int, 6> all_boundary_type_p = {2, 2, 2, 2, 1, 1};
        StokesFlow<DIM>::all_boundary_type_p   = all_boundary_type_p;
        StokesFlow<DIM>::all_boundary_type     = all_boundary_type;

        // 其他初始化操作
        StokesFlow<DIM>::initial_boundary_conditions();
        set_boundary_conditions();
    }

  private:
    /// @brief 设置初始值
    virtual void initial_values() final override {}

    /// @brief 设置边界值(可能会随时间变化)
    virtual void set_boundary_conditions() final override {}

    /// @brief 设置源项(可能会随时间变化)
    virtual void set_source_terms() final override {}

    /// @brief 后处理
    virtual void post_process(int i) final override {}
};

} // namespace stokes_flow
