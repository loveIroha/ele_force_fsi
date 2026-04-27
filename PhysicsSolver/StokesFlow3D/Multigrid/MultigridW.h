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
class MultigridW : public Multigrid<double, View3D> {
  private:
  public:
    MultigridW(int3 N, double3 L, View3D<int> bct) : Multigrid{N, L, bct} { pad.z = 3; }

    ~MultigridW() override {}
};
} // namespace mg