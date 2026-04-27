/**
 * @file BuildDofmap.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-12
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include "BuildDofmap.h"

#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/Vertex.h>
#include <io/loguru.hpp>

#include "LagrangeFirst.h"
#include "LagrangeSecond.h"

class Position : public dolfin::Expression {
  public:
    Position() : dolfin::Expression(3) {}

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

template <int degree>
struct FunctionSpace;

template <>
struct FunctionSpace<1> {
    static const int num_local_dofs = 4;
    using Type                      = LagrangeFirst::FunctionSpace;
};
template <>
struct FunctionSpace<2> {
    static const int num_local_dofs = 10;
    using Type                      = LagrangeSecond::FunctionSpace;
};

// 这个函数只会被BasicMesh类中的方法调用，使用BasicMesh的人不知道这个函数的存在，那么链接时应该使用private
template <int degree>
void dolfin_build_dof(const std::shared_ptr<dolfin::Mesh> mesh, std::vector<int4>& cells,
                      std::vector<double3>& vertices, std::vector<size_t>& dofmaps,
                      std::vector<double3>& dof_coordinates) {
    CHECK_F(degree == 1 || degree == 2, "degree must be 1 or 2");

    auto num_local_dofs = FunctionSpace<degree>::num_local_dofs;

    auto V      = std::make_shared<typename FunctionSpace<degree>::Type>(mesh);
    auto dofmap = V->dofmap();
    auto u      = dolfin::Function(V);

    Position expr;
    u.interpolate(expr);

    cells.resize(mesh->num_cells());
    dofmaps.resize(mesh->num_cells() * num_local_dofs);
    vertices.resize(mesh->num_vertices());
    dof_coordinates.resize(u.vector()->size() / 3);

    LOG_F(INFO, "mesh->num_cells() : %ld", mesh->num_cells());

    for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell) {
        auto& a = cells[cell->index()];
        auto  b = cell->entities(0);
        a.x     = b[0];
        a.y     = b[1];
        a.z     = b[2];
        a.w     = b[3];
    }

    for (dolfin::VertexIterator vertex(*mesh); !vertex.end(); ++vertex) {
        auto& v = vertices[vertex->index()];
        v.x     = vertex->point().x();
        v.y     = vertex->point().y();
        v.z     = vertex->point().z();
    }

    for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell) {
        auto local_dofmap = dofmap->cell_dofs(cell->index());
        // LOG_F(INFO, "degree %d, local_dofmap.size() %d, num_local_dofs %d",
        // degree, local_dofmap.size(), num_local_dofs);
        CHECK_F(local_dofmap.size() == num_local_dofs * 3, "Wrong size. it should be a 2nd Lagrangian function.");
        for (int i = 0; i < num_local_dofs; i++) {
            dofmaps[cell->index() * num_local_dofs + i] = (local_dofmap[i] / 3);
        }
    }

    for (size_t i = 0; i < u.vector()->size() / 3; i++) {
        double3 dof_coordinate;
        dof_coordinate.x   = u.vector()->getitem(3 * i);
        dof_coordinate.y   = u.vector()->getitem(3 * i + 1);
        dof_coordinate.z   = u.vector()->getitem(3 * i + 2);
        dof_coordinates[i] = dof_coordinate;
    }
}

template void dolfin_build_dof<2>(const std::shared_ptr<dolfin::Mesh> mesh, std::vector<int4>& cells,
                                  std::vector<double3>& vertices, std::vector<size_t>& dofmaps,
                                  std::vector<double3>& dof_coordinates);

template void dolfin_build_dof<1>(const std::shared_ptr<dolfin::Mesh> mesh, std::vector<int4>& cells,
                                  std::vector<double3>& vertices, std::vector<size_t>& dofmaps,
                                  std::vector<double3>& dof_coordinates);
