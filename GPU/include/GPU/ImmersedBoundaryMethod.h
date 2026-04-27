/**
 * @file ImmersedBoundaryMethod.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-07
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */
#ifndef __GPU_IMMERSED_BOUNDARY_METHOD_H__
#define __GPU_IMMERSED_BOUNDARY_METHOD_H__
#include <GPU/utilities.h>
#include <io/IBTimer.h>

namespace gpu {

// // T   : double or float
// // TV  : double, double3 or float, float3
// // TV4 : double4 or float4
// // The size of data type should be checked
// template<typename T, typename TV3, typename TV4>
// void distribute_force_cpu(
//     const TV3* solid_forces,
//     const TV4* quadrature_rules,
//           TV3* fluid_forces,
//     int num, double h, int3 dim);

void distribute_force_cpu(const double3* solid_forces,     /// solid_forces
                          const double4* quadrature_rules, /// quadrature quadrature_rules
                          double3* fluid_forces, int num, double h, int3 dim);

void interpolate_velocity_cpu(double3* solid_velocities, const double4* quadrature_rules,
                              const double3* fluid_velocities, int num, double h, int3 dim);

} // namespace gpu

#endif