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
class MultigridV : public Multigrid<double, View3D> {
  private:
  public:
    MultigridV(int3 N, double3 L, View3D<int> bct) : Multigrid{N, L, bct} { pad.y = 3; }

    ~MultigridV() override {}
};
} // namespace mg