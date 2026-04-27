/// @date 2023-12-18
/// @file MultigridU.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei & Wang Xuan
///
/// @brief
///
///

#pragma once

#include <config.h>

#include "Multigrid.h"

namespace mg {
class MultigridU : public Multigrid<double, View3D> {
  private:
  public:
    MultigridU(int3 N, double3 L, View3D<int> bct) : Multigrid{N, L, bct} {
        LOG_F(INFO, "Create MultigridU");
        pad.x = 3;
    }

    ~MultigridU() override {}

    virtual void restrict_bc(View3D<int>& coarse, const View3D<int>& fine, int3 N) override final {
        LOG_F(INFO, "MultigridU::restrict_bc");
        typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;
        Kokkos::parallel_for(
            "restrict_bc_1", mdrange_policy({0, 0}, {N.y, N.z}), KOKKOS_LAMBDA(const int j, const int k) {
                coarse(0, j + 1, k + 1) = max(max(fine(0, 2 * j + 1, 2 * k + 1), fine(0, 2 * j + 2, 2 * k + 1)),
                                              max(fine(0, 2 * j + 1, 2 * k + 2), fine(0, 2 * j + 2, 2 * k + 2)));
                coarse(N.x + 2, j + 1, k + 1)
                    = max(max(fine(2 * N.x + 2, 2 * j + 1, 2 * k + 1), fine(2 * N.x + 2, 2 * j + 2, 2 * k + 1)),
                          max(fine(2 * N.x + 2, 2 * j + 1, 2 * k + 2), fine(2 * N.x + 2, 2 * j + 2, 2 * k + 2)));
            });
        Kokkos::parallel_for(
            "restrict_bc_2", mdrange_policy({0, 0}, {N.x + 1, N.z}), KOKKOS_LAMBDA(const int i, const int k) {
                coarse(i + 1, 0, k + 1) = max(fine(2 * i + 1, 0, 2 * k + 1), fine(2 * i + 1, 0, 2 * k + 2));
                coarse(i + 1, N.y + 1, k + 1)
                    = max(fine(2 * i + 1, 2 * N.y + 1, 2 * k + 2), fine(2 * i + 1, 2 * N.y + 1, 2 * k + 2));
            });
        Kokkos::parallel_for(
            "restrict_bc_3", mdrange_policy({0, 0}, {N.x + 1, N.y}), KOKKOS_LAMBDA(const int i, const int j) {
                coarse(i + 1, j + 1, 0) = max(fine(2 * i + 1, 2 * j + 1, 0), fine(2 * i + 1, 2 * j + 2, 0));
                coarse(i + 1, j + 1, N.z + 1)
                    = max(fine(2 * i + 1, 2 * j + 1, 2 * N.z + 1), fine(2 * i + 1, 2 * j + 2, 2 * N.z + 1));
            });
    }
};
} // namespace mg