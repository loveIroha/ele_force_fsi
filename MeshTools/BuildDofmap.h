/**
 * @file BuildDofmap copy.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-12
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <memory>
// types such as int3 double3...
#include <dolfin/mesh/Mesh.h>
#include <helper_math.h>
#include <vector_types.h>

template <int degree = 2>
void dolfin_build_dof(const std::shared_ptr<dolfin::Mesh> mesh, std::vector<int4>& cells,
                      std::vector<double3>& vertices, std::vector<size_t>& dofmaps,
                      std::vector<double3>& dof_coordinates);