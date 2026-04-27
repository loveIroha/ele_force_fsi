/**
 * @file GenericSolidSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-03
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __GENERIC_SOLID_SOLVER_2D_H__
#define __GENERIC_SOLID_SOLVER_2D_H__

#include <AlgebraSolver/algebra.h>
#include <dolfin.h>
#include <io/loguru.hpp>
#include <vector_types.h>
namespace dolfin {

// TODO : remove it.
class TmpRefConfiguration : public Expression {
  public:
    TmpRefConfiguration() : Expression(2) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
    }
};

template <typename UserFunctionSpace, typename UserBilinearForm, typename UserLinearForm>
class GenericSolidSolver {
  public:
    double _t  = 0.0;
    double _dt = 0.0;

    Matrix A;
    Vector b;
    Vector mass;

    std::shared_ptr<UserLinearForm>    L;
    std::shared_ptr<UserBilinearForm>  a;
    std::shared_ptr<UserFunctionSpace> V;

    std::shared_ptr<Mesh> _mesh;

    // NOTE: there could be four types of markers points, lines, facets, cells
    std::shared_ptr<MeshFunction<size_t>> boundaries_points;
    std::shared_ptr<MeshFunction<size_t>> _boundaries;
    std::shared_ptr<MeshFunction<size_t>> _material_types;

    File dfile;
    File ffile;

  public:
    virtual ~GenericSolidSolver(){};

    GenericSolidSolver(std::shared_ptr<Mesh> mesh) : GenericSolidSolver(mesh, "") {}

    void constraint_body(std::shared_ptr<Function> F, const std::shared_ptr<Function> X,
                         const std::shared_ptr<Function> U, double kappa, const double eta, size_t marker) {
        auto dofmap = V->dofmap();
        // auto element = V->element();
        std::vector<double> F_vector(F->vector()->local_size());

        for (dolfin::CellIterator cell(*_mesh); !cell.end(); ++cell) {
            // 找到被标记的单元
            if ((*_material_types)[*cell] == marker) {
                // auto p = cell->midpoint();
                // LOG_F(INFO, "midpoint of cell marked as %zu :  %f %f %f .",
                // mark, p.x(), p.y(), p.z());

                // 找到被标记单元的 cell_dofmap
                // cell_dofmap[k] 表示此单元第 k 个 dof 在向量中的索引
                // 这里的向量是指 Function 中 vector() 指向的向量
                auto cell_dofmap = dofmap->cell_dofs(cell->index());

                // 输出坐标
                for (int k = 0; k < cell_dofmap.size(); k++) {
                    F_vector[cell_dofmap[k]]
                        = -(*X->vector())[cell_dofmap[k]] * kappa - (*U->vector())[cell_dofmap[k]] * eta;
                }
            }
        }
        F->vector()->set_local(F_vector);
    }

    void constraint_facets(std::shared_ptr<Function> F, const std::shared_ptr<Function> X,
                           const std::shared_ptr<Function> U, double kappa, const double eta, size_t marker) {
        // Extract boundary marker
        CHECK_F(_boundaries != nullptr, "_boundaries is a null pointer");
        const MeshFunction<size_t>& boundaries = *(_boundaries);

        // Extract mesh
        CHECK_F(_mesh != nullptr, "Mesh is a null pointer");
        const Mesh& mesh = *(_mesh);

        // Compute facets and facet - cell connectivity if not already computed
        const std::size_t D = mesh.topology().dim();
        mesh.init(D - 1);
        mesh.init(D - 1, D);
        CHECK_F(mesh.ordered(), "Mesh should be ordered.");

        // 查找并存储标记单元
        std::vector<std::size_t> marked_facet_indices;
        for (FacetIterator facet(mesh); !facet.end(); ++facet) {
            // 找到被标记的单元
            if (boundaries[*facet] == marker) { marked_facet_indices.push_back(facet->index()); }
        }

        // 查找标记单元对应的 dofmap
        // Collect pointers to dof maps. ( get(): get the raw pointer. )
        const GenericDofMap* dofmaps = V->dofmap().get();

        // Vector to hold dof map for a cell
        auto dofs = dofmaps->entity_closure_dofs(mesh, D - 1, marked_facet_indices);

        // 计算表面力，将其存入数组
        std::vector<double> F_vector(F->vector()->local_size());
        for (std::size_t dof : dofs) {
            F_vector[dof] = -(*X->vector())[dof] * kappa - (*U->vector())[dof] * eta;
        }
        F->vector()->set_local(F_vector);
    }

    // void find_marked_cell(size_t mark)
    // {
    //     auto dofmap = V->dofmap();
    //     auto element = V->element();
    //
    //     TmpRefConfiguration trc;
    //     auto reference_configuration = std::make_shared<Function>(V);
    //     reference_configuration->interpolate(trc);
    //     auto &trc_vector = *(reference_configuration->vector());
    //
    //     CHECK_F(_material_types != nullptr, "_material_types is nullptr");
    //     for (dolfin::CellIterator cell(*_mesh); !cell.end(); ++cell)
    //     {
    //         // 找到被标记的单元
    //         if ((*_material_types)[*cell] == mark)
    //         {
    //             auto p = cell->midpoint();
    //             LOG_F(INFO, "midpoint of cell marked as %zu :  %f %f %f .",
    //             mark, p.x(), p.y(), p.z());
    //
    //             // 找到被标记单元的 cell_dofmap
    //             // cell_dofmap[k] 表示此单元第 k 个 dof 在向量中的索引
    //             // 这里的向量是指 Function 中 vector() 指向的向量
    //             auto cell_dofmap = dofmap->cell_dofs(cell->index());
    //
    //             // 计算被标记单元每个 dof 对应的坐标
    //             // P1 元有三个 dof ，每个 dof 对应一个坐标
    //             std::vector<double> coordinate_dofs;
    //             cell->get_coordinate_dofs(coordinate_dofs);
    //             boost::multi_array<double, 2> dof_coordinates;
    //             element->tabulate_dof_coordinates(dof_coordinates,
    //             coordinate_dofs, *cell);
    //
    //             // 输出坐标
    //             for (size_t k = 0; k < cell_dofmap.size(); k++)
    //             {
    //                 LOG_F(INFO, "dof index: %d, dof value: %f,
    //                 dof_coordinates: %f %f .", cell_dofmap[k],
    //                 trc_vector[cell_dofmap[k]], dof_coordinates[k][0],
    //                 dof_coordinates[k][1]);
    //             }
    //         }
    //     }
    // }

    GenericSolidSolver(std::shared_ptr<Mesh> mesh, std::string result_path)
        : L(nullptr), a(nullptr), V(nullptr), _mesh(mesh), boundaries_points(nullptr), _boundaries(nullptr),
          _material_types(nullptr), dfile(result_path + "solid/position.pvd"), ffile(result_path + "solid/force.pvd") {
        LOG_F(INFO, " GenericSolidSolver is called!");

        // Define function space, variational forms and MeshFunction (boundary
        // faces)
        V = std::make_shared<UserFunctionSpace>(_mesh);
        a = std::make_shared<UserBilinearForm>(V, V);
        L = std::make_shared<UserLinearForm>(V);

        // Assemble matrix A
        assemble(A, *a);
    }

    std::shared_ptr<UserFunctionSpace> function_space() const { return V; }

    template <typename TV, typename T>
    void record(const std::vector<TV>& G_v, const std::vector<TV>& X_v, double t) {
        auto G = std::make_shared<Function>(V);
        auto X = std::make_shared<Function>(V);

        G->vector()->set_local(algebra::flatten<TV, T>(G_v));
        X->vector()->set_local(algebra::flatten<TV, T>(X_v));
        dfile.write(*X, t);
        ffile.write(*G, t);
    }
};
} // namespace dolfin
#endif
