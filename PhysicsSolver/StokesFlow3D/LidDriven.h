/// @date 2023-10-22
/// @file LidDriven.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include "Kokkos/StokesFlow3D_kokkos.h"

namespace stokes_flow {
template <int DIM>
class LidDriven : public StokesFlow<DIM> {
  public:
    LidDriven(std::shared_ptr<NavierStokesDemo> _ns_demo, std::string _output_file)
        : StokesFlow<DIM>(_ns_demo, _output_file + "fluid/data.pvd") {
        // 设置边界类型和边界值
        std::array<int, 6> all_boundary_type   = {1, 1, 1, 1, 1, 1};
        std::array<int, 6> all_boundary_type_p = {2, 2, 2, 2, 2, 2};
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
    virtual void set_boundary_conditions() final override {
        for (int i = 1; i < StokesFlow<DIM>::_Nx; i++) {
            for (int k = 0; k < StokesFlow<DIM>::_Nz; k++) {
                StokesFlow<DIM>::ubv[i + 1][StokesFlow<DIM>::_Ny + 1][k + 1] = 1.0;
            }
        }
    }

    /// @brief 设置源项(可能会随时间变化)
    virtual void set_source_terms() final override {}

    /// @brief 后处理
    virtual void post_process(int i) final override {}
};

} // namespace stokes_flow
