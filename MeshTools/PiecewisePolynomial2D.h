/// @date 2023-06-14
/// @file PiecewisePolynomial.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef _PIECEWISE_POLYNOMIAL_2D_H_
#define _PIECEWISE_POLYNOMIAL_2D_H_

/// CUDA header
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <vector_types.h>

/// my gpu library
#include <GPU/gpu_lib.h>

// Loguru
#include <io/loguru.hpp>

// 积分点和高斯积分权重
#include <MeshTools/QuadratureRules.h>

template <int degree = 2, int quadrature_order = 3, int DIM = 2>
class PiecewisePolynomial {
  private:
    // Geometry information for reference cell.
    // static const double3 tetrahedron_vertices[4];
    // static const double3 triangle_vertices[3];

    // static const double3 quadratic_dof_coordinates[10];
    // static const double3 linear_dof_coordinates[4];

    // Gauss quadrature rules.
    static const npuheart::SimplexQuadrature<double3, double, quadrature_order, DIM> simplex_quadrature;

  public:
    static constexpr size_t num_gauss_points() { return simplex_quadrature.num_points(); }
    constexpr int           num_local_dofs() const {
        if constexpr (degree == 1 && DIM == 2) return 3;
        if constexpr (degree == 2 && DIM == 2) return 6;
        if constexpr (degree == 1 && DIM == 3) return 4;
        if constexpr (degree == 2 && DIM == 3)
            return 10;
        else
            return -1;
    }

  public:
    PiecewisePolynomial() {}

    ~PiecewisePolynomial() {}

    // 输入三个顶点，输出
    void get_transformation_operator(const double3& p1, const double3& p2, const double3& p3, double* H, double* b,
                                     double* inv_H, double* inv_Hb) {
        H[0] = p2.x - p1.x;
        H[1] = p3.x - p1.x;

        H[2] = p2.y - p1.y;
        H[3] = p3.y - p1.y;

        double det_H = H[0] * H[3] - H[1] * H[2];

        inv_H[0] = H[3] / det_H;
        inv_H[1] = H[2] / det_H;
        inv_H[2] = H[1] / det_H;
        inv_H[3] = H[0] / det_H;

        b[0] = p1.x;
        b[1] = p1.y;

        inv_Hb[0] = -inv_H[0] * b[0] - inv_H[1] * b[1];
        inv_Hb[1] = -inv_H[2] * b[0] - inv_H[3] * b[1];
    }

    void get_transformation_operator_all_cells(const int4* cells, const double3* vertices, double* Hs, double* bs,
                                               double* inv_Hs, double* inv_Hbs, size_t num_cells) {
        // calculate quadrature rules on local cell.
        for (size_t index = 0; index < num_cells; index++) {
            const double3& p1 = vertices[cells[index].x];
            const double3& p2 = vertices[cells[index].y];
            const double3& p3 = vertices[cells[index].z];

            auto H      = &(Hs[DIM * DIM * index]);
            auto b      = &(bs[DIM * index]);
            auto inv_H  = &(inv_Hs[DIM * DIM * index]);
            auto inv_Hb = &(inv_Hbs[DIM * index]);

            get_transformation_operator(p1, p2, p3, H, b, inv_H, inv_Hb);
        }
    }

    // x'=Ax+b
    void transform_a_point(double* point_out, const double* point_in, const double* A, const double* b) {
        // for (size_t i = 0; i < DIM; i++)
        // {
        //     for (size_t j = 0; j < DIM; j++)
        //     {
        //         point_out[i] = A[i * DIM + j] * point_in[j];
        //     }
        //     point_out[i] += b[i];
        // }
        for (size_t i = 0; i < DIM; i++) {
            point_out[i] = 0.0;
            for (size_t j = 0; j < DIM; j++)
                point_out[i] += A[i * DIM + j] * point_in[j];
            point_out[i] += b[i];
        }
        // point_out[0] = A[0 * DIM + 0] * point_in[0] + A[0 * DIM + 1] *
        // point_in[1] + b[0]; point_out[1] = A[1 * DIM + 0] * point_in[0] + A[1 *
        // DIM + 1] * point_in[1] + b[1];
    }

    void transform_a_point(double3& point_out, const double3& point_in, const double* A, const double* b) {
        double* point_out_ptr = (double*)&point_out;
        double* point_in_ptr  = (double*)&point_in;
        transform_a_point(point_out_ptr, point_in_ptr, A, b);
    }

    void get_local_quadrature_rules(const double* Hs, const double* bs, const double* volumes,
                                    double4* quadrature_rules, size_t num_cells) {
        //
        for (size_t index = 0; index < num_cells; index++) {
            const double* H = &(Hs[index * DIM * DIM]);
            const double* b = &(bs[index * DIM]);

            for (size_t i = 0; i < num_gauss_points(); i++) {
                // Transform reference quadrature point to local quadrature point
                double3 point_local{};
                double3 point_ref = simplex_quadrature._points[i];
                transform_a_point(point_local, point_ref, H, b);
                // printf("%d\n", i);
                // print_double3(point_local, "point_local");
                // print_double3(point_ref, "point_ref");

                // Write to quadrature rules
                auto qr_index              = index * num_gauss_points() + i;
                quadrature_rules[qr_index] = make_double4(point_local.x, point_local.y, point_local.z,
                                                          simplex_quadrature._weights[i] * 2.0 * volumes[index]);
            }
        }
    }

    double triangle_area(const double3* points) {
        const double* x0 = (double*)&(points[0]);
        const double* x1 = (double*)&(points[1]);
        const double* x2 = (double*)&(points[2]);

        // Compute area of triangle embedded in R^3
        // const double v0 = (x0[1] * x1[2] + x0[2] * x2[1] + x1[1] * x2[2]) -
        // (x2[1] * x1[2] + x2[2] * x0[1] + x1[1] * x0[2]); const double v1 = (x0[2]
        // * x1[0] + x0[0] * x2[2] + x1[2] * x2[0]) - (x2[2] * x1[0] + x2[0] * x0[2]
        // + x1[2] * x0[0]); const double v2 = (x0[0] * x1[1] + x0[1] * x2[0] +
        // x1[0] * x2[1]) - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]); return
        // 0.5 * sqrt(v0 * v0 + v1 * v1 + v2 * v2);

        // Compute area of triangle embedded in R^2
        double v2 = (x0[0] * x1[1] + x0[1] * x2[0] + x1[0] * x2[1]) - (x2[0] * x1[1] + x2[1] * x0[0] + x1[0] * x0[1]);
        return 0.5 * std::abs(v2);
    }

    void print_double3(double3 data, const std::string& name) {
        LOG_F(INFO, "%s : %f %f %f. ", name.c_str(), data.x, data.y, data.z);
    }

    void triangle_area_all_cells(const int4* cells, const double3* vertices, double* volumes, size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            const double3 points[3] = {vertices[cells[index].x], vertices[cells[index].y], vertices[cells[index].z]};

            volumes[index] = triangle_area(points);
        }
    }

    void print_double2(double2 data, const std::string& name) {
        LOG_F(INFO, "%s : %f %f. ", name.c_str(), data.x, data.y);
    }

    void print_double4(double4 data, const std::string& name) {
        LOG_F(INFO, "%s : %f %f %f %f. ", name.c_str(), data.x, data.y, data.z, data.w);
    }

    template <typename TV, typename T>
    void evaluate(const TV* dofs, const double3* points, TV* results, size_t num) {
        if (degree == 1) {
            evaluate_linear<TV, T>(dofs, points, results, num);
        } else if (degree == 2) {
            evaluate_quadratic<TV, T>(dofs, points, results, num);
        } else {
            LOG_F(ERROR, "degree %d is not supported.", degree);
        }
    }

    template <typename TV, typename T>
    void evaluate_linear(const TV* dofs, const double3* points, TV* results, size_t num) {
        // LOG_F(INFO, "evaluate_quadratic");
        // LOG_F(INFO,"dofs: %f %f %f", dofs[0].x, dofs[1].x, dofs[2].x);
        // LOG_F(INFO, "dofs: %f %f %f", dofs[0].y, dofs[1].y, dofs[2].y);
        for (size_t i = 0; i < num; i++) {
            double x = points[i].x;
            double y = points[i].y;

            results[i] = dofs[0] * (1 - x - y) // base 1
                         + dofs[1] * x         // base 2
                         + dofs[2] * y;        // base 3
        }
    }

    template <typename TV, typename T>
    void evaluate_quadratic(const TV* dofs, const double3* points, TV* results, size_t num) {
        // LOG_F(INFO, "evaluate_quadratic");
        // LOG_F(INFO,"dofs: %f %f %f %f %f %f", dofs[0].x, dofs[1].x, dofs[2].x,
        // dofs[3].x, dofs[4].x, dofs[5].x); LOG_F(INFO, "dofs: %f %f %f %f %f %f",
        // dofs[0].y, dofs[1].y, dofs[2].y, dofs[3].y, dofs[4].y, dofs[5].y);
        for (size_t i = 0; i < num; i++) {
            double x = points[i].x;
            double y = points[i].y;

            results[i] = dofs[0] * (2.0 * x * x + 2.0 * y * y + 4.0 * x * y - 3.0 * y - 3.0 * x + 1) // base 1
                         + dofs[1] * (2.0 * x * x - x)                                               // base 2
                         + dofs[2] * (2.0 * y * y - y)                                               // base 3
                         + dofs[3] * (4.0 * x * y)                                                   // base 5
                         + dofs[4] * (-4.0 * y * y - 4.0 * x * y + 4.0 * y)                          // base 6
                         + dofs[5] * (-4.0 * x * x - 4.0 * x * y + 4.0 * x);                         // base 4
        }
    }

        template <typename TV, typename T>
    void evaluate_vertices(const TV* function, const size_t* dofmap, const int4* cells, const double3* vertices,
                           TV* results, size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            const double3 local_points[3] = {{0, 0}, {1, 0}, {0, 1}};

            TV local_results[3];

            // LOG_F(INFO, "local_points: %f %f %f", local_points[0].x,
            // local_points[0].y, local_points[0].z); LOG_F(INFO, "local_points: %f
            // %f %f", local_points[1].x, local_points[1].y, local_points[1].z);
            // LOG_F(INFO, "local_points: %f %f %f", local_points[2].x,
            // local_points[2].y, local_points[2].z);

            // Get local dofs from function and dofmap
            TV dof[num_local_dofs()];
            for (int i = 0; i < num_local_dofs(); i++) {
                dof[i] = function[dofmap[index * num_local_dofs() + i]];
            }

            // Evaluate at quadrature points
            evaluate<TV, T>(dof, local_points, local_results, 3);

            results[cells[index].x] = local_results[0];
            results[cells[index].y] = local_results[1];
            results[cells[index].z] = local_results[2];
        }
    }


    template <typename TV, typename T>
    void evaluate_quadrature_points(const TV* function, const size_t* dofmap, TV* results, size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            TV* result = &(results[num_gauss_points() * index]);

            // Get local dofs from function and dofmap
            TV dof[num_local_dofs()];
            for (int i = 0; i < num_local_dofs(); i++) {
                dof[i] = function[dofmap[index * num_local_dofs() + i]];
            }

            // Evaluate at quadrature points
            evaluate<TV, T>(dof, simplex_quadrature._points, result, num_gauss_points());
        }
    }

    // evaluate_quadrature_points
    std::vector<double> calculate_basis_values() {
        // Evaluate at quadrature points
        static std::vector<double> results(num_local_dofs() * num_gauss_points());
        static bool                calculated = false;
        if (calculated) return results;

        for (int i = 0; i < num_local_dofs(); i++) {
            std::vector<double> basis_dofs(num_local_dofs());
            basis_dofs[i]  = 1.0;
            auto results_i = &(results.data()[i * num_gauss_points()]);

            evaluate<double, double>(basis_dofs.data(), simplex_quadrature._points, results_i, num_gauss_points());
        }
        // LOG_F(INFO, "basis values calculating...");

        auto data1 = results;
        for (size_t logf_i = 0; logf_i < data1.size(); logf_i++) {
            LOG_F(INFO, "basis values on gauss quadrature points: %zu %.16e.", logf_i, data1[logf_i]);
        }
        calculated = true;
        return results;
    }
};

template <int degree, int quadrature_order, int DIM>
const npuheart::SimplexQuadrature<double3, double, quadrature_order, DIM>
    PiecewisePolynomial<degree, quadrature_order, DIM>::simplex_quadrature
    = npuheart::simplex_quadrature<double3, double, quadrature_order, DIM>;

#endif