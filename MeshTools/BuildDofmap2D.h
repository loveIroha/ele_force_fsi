// @date 2023-06-14
/// @file BuildDofmap2D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 加入二维的代码。利用dolfin工具生成网格数据和有限元dofmap
///        2023/9/4 二维二阶代码
///

#ifndef __BUILD_DOFMAP_2D_H__
#define __BUILD_DOFMAP_2D_H__

// Dolfin
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/Vertex.h>

// ufl
#include "LagrangeFirst.h"    // 三维 一阶
#include "LagrangeFirst2D.h"  // 二维 一阶
#include "LagrangeSecond.h"   // 三维 二阶
#include "LagrangeSecond2D.h" // 二维 二阶

//
#include <io/loguru.hpp>

#include "BuildDofmap.h"

template <int DIM>
class Position : public dolfin::Expression {
  public:
    Position() : dolfin::Expression(DIM) {}

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const {
        for (int i = 0; i < DIM; i++)
            values[i] = x[i];
    }
};

template <int DEGREE, int DIM>
struct FunctionSpace;

template <>
struct FunctionSpace<1, 2> {
    static const int num_local_dofs = 3;
    using Type                      = LagrangeFirst2D::FunctionSpace;
};

template <>
struct FunctionSpace<2, 2> {
    static const int num_local_dofs = 6;
    using Type                      = LagrangeSecond2D::FunctionSpace;
};
template <>
struct FunctionSpace<1, 3> {
    static const int num_local_dofs = 4;
    using Type                      = LagrangeFirst::FunctionSpace;
};
template <>
struct FunctionSpace<2, 3> {
    static const int num_local_dofs = 10;
    using Type                      = LagrangeSecond::FunctionSpace;
};

// 这个函数只会被BasicMesh类中的方法调用，使用BasicMesh的人不知道这个函数的存在，那么链接时应该使用private
template <int DEGREE, int DIM>
void dolfin_build_dof(const std::shared_ptr<dolfin::Mesh> mesh, std::vector<int4>& cells,
                      std::vector<double3>& vertices, std::vector<size_t>& dofmaps,
                      std::vector<double3>& dof_coordinates) {
    CHECK_F(DEGREE == 1 || DEGREE == 2, "DEGREE must be 1 or 2");
    CHECK_F(DIM == 2 || DIM == 3, "DIM must be 1 or 2 or 3");

    auto num_local_dofs = FunctionSpace<DEGREE, DIM>::num_local_dofs;

    auto V      = std::make_shared<typename FunctionSpace<DEGREE, DIM>::Type>(mesh);
    auto dofmap = V->dofmap();
    auto u      = dolfin::Function(V);

    Position<DIM> expr;
    u.interpolate(expr);

    cells.resize(mesh->num_cells());
    dofmaps.resize(mesh->num_cells() * num_local_dofs);
    vertices.resize(mesh->num_vertices());
    dof_coordinates.resize(u.vector()->size() / DIM);

    // LOG_F(INFO, "mesh->num_cells() : %ld", mesh->num_cells());

    // 单元
    for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell) {
        auto a = (int*)&cells[cell->index()];
        auto b = cell->entities(0);
        // NOTE: 只适合单纯形
        for (int i = 0; i < DIM + 1; i++)
            a[i] = b[i];
    }

    // 节点
    for (dolfin::VertexIterator vertex(*mesh); !vertex.end(); ++vertex) {
        auto& v = vertices[vertex->index()];
        v.x     = vertex->point().x();
        v.y     = vertex->point().y();
        v.z     = vertex->point().z();
    }

    // dofmap
    for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell) {
        auto local_dofmap = dofmap->cell_dofs(cell->index());
        // LOG_F(INFO, "DEGREE %d, local_dofmap.size() %d, num_local_dofs %d",
        // DEGREE, local_dofmap.size(), num_local_dofs);
        CHECK_F(local_dofmap.size() == num_local_dofs * DIM, "Wrong size. it should be a 2nd Lagrangian function.");
        for (int i = 0; i < num_local_dofs; i++) {
            dofmaps[cell->index() * num_local_dofs + i] = (local_dofmap[i] / DIM);
        }
    }

    // dof_coordinates
    for (size_t i = 0; i < u.vector()->size() / DIM; i++) {
        auto dof_coordinate = (double*)&dof_coordinates[i];
        for (int j = 0; j < DIM; j++) {
            dof_coordinate[j] = u.vector()->getitem(DIM * i + j);
        }
    }
}

// template <int degree = 2, int DIM = 3>
// void dolfin_build_dof(const std::shared_ptr<dolfin::Mesh> mesh,
//                       std::vector<int4> &cells,
//                       std::vector<double3> &vertices,
//                       std::vector<size_t> &dofmaps,
//                       std::vector<double3> &dof_coordinates);

#endif