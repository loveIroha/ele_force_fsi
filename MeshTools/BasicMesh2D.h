/**
 * @file BasicMesh.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-11-30
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _BASIC_MESH_H_
#define _BASIC_MESH_H_

#include <boost/any.hpp>

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

/// CUDA header
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <vector_types.h>

/// my classes
#include <AlgebraSolver/GpuVector.h>
#include <AlgebraSolver/algebra.h>

/// loguru
#include <MeshTools/BuildDofmap2D.h>
#include <MeshTools/MeshFunctionManager.h>
#include <MeshTools/PiecewisePolynomial2D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/DistributionInterpolation2D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/DistributionInterpolation2D_kokkos.h>
#include <config.h>
#include <io/loguru.hpp>

template <int DEGREE, int DIM>
class BasicMesh : public MeshFunctionManager {
  public:
    std::shared_ptr<dolfin::Mesh> _mesh;

    // Coordinates for all dofs.
    std::vector<double3> vertices;
    std::vector<int4>    cells;
    std::vector<int3>    boundaries;

    // used for quadratic function defined on the mesh.
    std::vector<size_t>  dofmaps;
    std::vector<double3> dof_coordinates;

    PiecewisePolynomial<DEGREE, 5, DIM> polynomial;

    // transformation operator, calculated only once.
    std::vector<double> Hs;
    std::vector<double> inv_Hs;
    std::vector<double> bs;
    std::vector<double> inv_Hbs;

    // volumes and quadrature rules, calculated only once.
    std::vector<double>  volumes;
    std::vector<double4> quadrature_rules;

  public:
    BasicMesh(std::shared_ptr<dolfin::Mesh> mesh) : _mesh{mesh}, polynomial{} {
        CHECK_F(_mesh != nullptr, "Empty mesh!");

        dolfin_build_dof<DEGREE, DIM>(_mesh, cells, vertices, dofmaps, dof_coordinates);

        Hs.resize(DIM * DIM * num_cells());
        bs.resize(DIM * num_cells());
        inv_Hs.resize(DIM * DIM * num_cells());
        inv_Hbs.resize(DIM * num_cells());
        volumes.resize(num_cells());
        quadrature_rules.resize(polynomial.num_gauss_points() * num_cells());

        // 计算体积
        polynomial.triangle_area_all_cells(cells.data(), vertices.data(), volumes.data(), num_cells());

        // 计算仿射变换算子
        polynomial.get_transformation_operator_all_cells(cells.data(), vertices.data(), Hs.data(), bs.data(),
                                                         inv_Hs.data(), inv_Hbs.data(), num_cells());

        // 计算高斯积分法则
        polynomial.get_local_quadrature_rules(Hs.data(), bs.data(), volumes.data(), quadrature_rules.data(),
                                              num_cells());

        // LOG_F(INFO, "The number of cells %ld \n\n", num_cells());
    }

    virtual ~BasicMesh() {}

    // number of cells
    size_t num_cells() const { return cells.size(); }

    /// Number of unkonwns for a scalar function in a cell. for quadratic
    /// element, it return 10.
    size_t num_local_dofs() const { return polynomial.num_local_dofs(); }

    // TODO : dof_coordinates is neccessary
    int num_dofs() const { return dof_coordinates.size(); }

    // 两个 set_function 能不能合并？
    std::vector<double2>& set_function(std::string name, Double2FunctionType function_expression) {
        // Check if data already exists
        auto it = functions_data.find(name);

        if (it == functions_data.end()) {
            LOG_F(INFO, "Function data named \"%s\" don't exists so we create one.", name.c_str());
            create_function<double2>(name);
            it = functions_data.find(name);
        }

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double2>&>(it->second);

        // Set function data
        function_data.resize(dof_coordinates.size());

        for (size_t i = 0; i < dof_coordinates.size(); i++) {
            function_data[i] = function_expression(dof_coordinates[i]);
            // LOG_F(INFO, "Dof coordinates %s[%ld] = (%f, %f)", name.c_str(),
            // i, dof_coordinates[i].x, dof_coordinates[i].y); LOG_F(INFO,
            // "Function data %s[%ld] = (%f, %f)", name.c_str(), i,
            // function_data[i].x, function_data[i].y);
        }

        return boost::any_cast<std::vector<double2>&>(it->second);
    }

    std::vector<double>& set_function(std::string name, ScalarFunctionType function_expression) {
        // Check if data already exists
        auto it = functions_data.find(name);

        if (it == functions_data.end()) {
            LOG_F(INFO, "Function data named \"%s\" don't exists so we create one.", name.c_str());
            create_function<double>(name);
            it = functions_data.find(name);
        }

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double>&>(it->second);

        // Set function data
        function_data.resize(dof_coordinates.size());

        for (size_t i = 0; i < dof_coordinates.size(); i++) {
            function_data[i] = function_expression(dof_coordinates[i]);
        }

        return boost::any_cast<std::vector<double>&>(it->second);
    }

    void distribute_source(std::vector<double>& eulerian_s, const std::vector<double>& lagrange_s,
                           const std::vector<double4>& quadrature_rules, double2 h, int2 dim_p) {
        std::fill(eulerian_s.data(), eulerian_s.data() + eulerian_s.size(), 0);
        distribute_s(lagrange_s.data(), eulerian_s.data(), quadrature_rules.size(), dim_p, h.x, h.y,
                     quadrature_rules.data());
    }

    void distribute_force(std::vector<double>& eulerian_u, std::vector<double>& eulerian_v,
                          const std::vector<double2>& lagrange, const std::vector<double4>& quadrature_rules, double2 h,
                          int2 dim_u, int2 dim_v) {
        auto [lagrange_u, lagrange_v] = algebra::split(lagrange);
        if constexpr (USE_KOKKOS) {
            distribute_u(eulerian_u.data(), lagrange_u.data(), quadrature_rules.data(), h, dim_u,
                         quadrature_rules.size());
            distribute_v(eulerian_v.data(), lagrange_v.data(), quadrature_rules.data(), h, dim_v,
                         quadrature_rules.size());
        } else {
            std::fill(eulerian_u.data(), eulerian_u.data() + eulerian_u.size(), 0);
            std::fill(eulerian_v.data(), eulerian_v.data() + eulerian_v.size(), 0);
            distribute_u(lagrange_u.data(), eulerian_u.data(), quadrature_rules.size(), dim_u, h.x, h.y,
                         quadrature_rules.data());

            distribute_v(lagrange_v.data(), eulerian_v.data(), quadrature_rules.size(), dim_v, h.x, h.y,
                         quadrature_rules.data());
        }
    }

    void interpolate_velocity(const std::vector<double>& eulerian_u, const std::vector<double>& eulerian_v,
                              std::vector<double2>& lagrange, const std::vector<double4>& quadrature_rules, double2 h,
                              int2 dim_u, int2 dim_v) {
        auto [lagrange_u, lagrange_v] = algebra::split(lagrange);
        if constexpr (USE_KOKKOS) {
            interpolate_u(lagrange_u.data(), eulerian_u.data(), quadrature_rules.data(), h, dim_u,
                          quadrature_rules.size());
            interpolate_v(lagrange_v.data(), eulerian_v.data(), quadrature_rules.data(), h, dim_v,
                          quadrature_rules.size());
        } else {
            interpolate_u(lagrange_u.data(), eulerian_u.data(), quadrature_rules.size(), dim_u, h.x, h.y,
                          quadrature_rules.data());
            interpolate_v(lagrange_v.data(), eulerian_v.data(), quadrature_rules.size(), dim_v, h.x, h.y,
                          quadrature_rules.data());
        }

        lagrange = algebra::merge(lagrange_u, lagrange_v);
    }

    template <typename TV, typename T>
    TV quadrature(std::string name) {
        TV              result{};
        std::vector<TV> values(quadrature_rules.size());
        auto&           function  = find_function<TV>(name);
        size_t          num_gauss = polynomial.num_gauss_points();
        polynomial.template evaluate_quadrature_points<TV, T>(function.data(), dofmaps.data(), values.data(),
                                                              num_cells());

        for (size_t i = 0; i < num_cells(); i++) {
            // auto local_values = &(values[num_gauss * i]);
            const TV*      local_values = &(values[num_gauss * i]);
            const double4* local_qr     = &quadrature_rules[num_gauss * i];
            for (size_t k = 0; k < num_gauss; k++) {
                result = result + local_qr[k].w * local_values[k];
            }
        }
        return result;
    }

    template <typename TV>
    void assemble_rhs_with_values_on_quadrature(const std::vector<TV>& values, std::vector<TV>& results) {
        auto num_gauss      = polynomial.num_gauss_points();
        auto basis_values   = polynomial.calculate_basis_values();
        int  num_local_dofs = polynomial.num_local_dofs();
        std::fill(results.begin(), results.end(), TV{});

        for (size_t i = 0; i < num_cells(); i++) {
            /// Calculate gauss points and values on them.
            auto local_values = &(values[num_gauss * i]);
            auto local_qr     = &quadrature_rules[num_gauss * i];

            for (int j = 0; j < num_local_dofs; j++) {
                // Evaluate basis values on a cell.
                size_t dof_index = dofmaps[num_local_dofs * i + j];
                for (size_t k = 0; k < num_gauss; k++) {
                    results[dof_index]
                        = results[dof_index] + local_qr[k].w * basis_values[j * num_gauss + k] * local_values[k];
                } // end the iteration of gauss quadrature points.
            }     // end the iteration of local dofs.
        }         // end the iteration of all cells.
    }
};

#endif
