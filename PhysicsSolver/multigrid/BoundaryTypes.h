/// @date 2023-05-18
/// @file BoundaryTypes.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __MULTIGRID_BOUNDARY_TYPES_H__
#define __MULTIGRID_BOUNDARY_TYPES_H__

namespace pangu {

enum Dimension { DIM1 = 1, DIM2 = 2, DIM3 = 3 };

const char INTERIOR  = 0;
const char DIRICHLET = 1;
const char NEUMANN   = 2;

const double pi = 3.14159265358979;

} // namespace pangu

#endif