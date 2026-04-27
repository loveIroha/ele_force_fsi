/**
 * @file PiecewisePolynomial.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-01
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _PIECEWISE_POLYNOMIAL_H_
#define _PIECEWISE_POLYNOMIAL_H_

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

// template <int degree = 2, int quadrature_order = 3>
// class PiecewisePolynomial;

template <int degree = 2>
class PiecewisePolynomial {
  private:
    // Geometry information for reference cell.
    static const double3 tetrahedron_vertices[4];
    static const double3 quadratic_dof_coordinates[10];
    static const double3 linear_dof_coordinates[4];

    // Gauss quadrature rules.
    static const double3 points_3[27];
    static const double  weights_3[27];
    static const double3 surface_points_3[27];          // 四面体单元表面上的高斯积分点
    static const double  surface_weights_3[27];         // 四面体单元表面上的高斯积分权重
    static const double  quadratic_basis_values_3[270]; // 二次节点基函数在高斯积分点处的值
    static const double  linear_basis_values_3[27 * 4]; // 一次节点基函数在高斯积分点处的值

    // NOTE : once it is changed, memory should be reallocated.
    size_t quadrature_order = 3;
    size_t num_gauss        = 27;

    // TODO : variable types are not consistent.
    double* quadrature_points  = nullptr;
    double* quadrature_weights = nullptr;

    // TODO : I think use constant memory will be much faster!
    double3* tetrahedron_vertices_device      = nullptr;
    double3* quadratic_dof_coordinates_device = nullptr;
    double3* quadrature_points_device         = nullptr;
    double*  quadrature_weights_device        = nullptr;
    double*  basis_values_device              = nullptr;

    bool use_gpu = false;

  public:
    const double* basis_values = nullptr;

    size_t num_gauss_points() {
        CHECK_F(quadrature_order == 3, "Wrong quadrature order.");
        if (quadrature_order == 3) return 27;
        return 0;
    }

    constexpr int num_local_dofs();

    void copy_variables_to_gpu() {
        CHECK_F(quadrature_order == 3, "Wrong quadrature order.");
        std::cout << "Allocating memory for PiecewisePolynomial object...." << std::endl;

        // const double3 *temp_points;
        // if (quadrature_order == 3) temp_points = points_3;

        // const double *temp_weights;
        // if (quadrature_order == 3) temp_weights = weights_3;

        if (tetrahedron_vertices_device == nullptr) gpu::freeGPUBuffer(tetrahedron_vertices_device);
        if (quadratic_dof_coordinates_device == nullptr) gpu::freeGPUBuffer(quadratic_dof_coordinates_device);
        if (quadrature_points_device == nullptr) gpu::freeGPUBuffer(quadrature_points_device);
        if (quadrature_weights_device == nullptr) gpu::freeGPUBuffer(quadrature_weights_device);
        if (basis_values_device == nullptr) gpu::freeGPUBuffer(basis_values_device);

        gpu::gpu_malloc((void**)&tetrahedron_vertices_device, sizeof(double3) * 4);
        gpu::gpu_malloc((void**)&quadratic_dof_coordinates_device, sizeof(double3) * 10);
        gpu::gpu_malloc((void**)&quadrature_points_device, sizeof(double3) * num_gauss_points());
        gpu::gpu_malloc((void**)&quadrature_weights_device, sizeof(double) * num_gauss_points());
        gpu::gpu_malloc((void**)&basis_values_device, sizeof(double) * 10 * num_gauss_points());

        gpu::cpu_to_gpu((char*)tetrahedron_vertices_device, (char*)tetrahedron_vertices, sizeof(double3) * 4);
        gpu::cpu_to_gpu((char*)quadratic_dof_coordinates_device, (char*)quadratic_dof_coordinates,
                        sizeof(double3) * 10);
        gpu::cpu_to_gpu((char*)quadrature_points_device, (char*)quadrature_points,
                        sizeof(double3) * num_gauss_points());
        gpu::cpu_to_gpu((char*)quadrature_weights_device, (char*)quadrature_weights,
                        sizeof(double) * num_gauss_points());
        gpu::cpu_to_gpu((char*)basis_values_device, (char*)basis_values, sizeof(double) * 10 * num_gauss_points());
    }

  public:
    PiecewisePolynomial() {
        CHECK_F(quadrature_order == 3, "Wrong quadrature order.");

        if (quadrature_order == 3) quadrature_points = (double*)points_3;
        if (quadrature_order == 3) quadrature_weights = (double*)weights_3;
        if (quadrature_order == 3 && degree == 2) basis_values = quadratic_basis_values_3;
        if (quadrature_order == 3 && degree == 1) basis_values = linear_basis_values_3;
        if (use_gpu == true) copy_variables_to_gpu();
        num_gauss = num_gauss_points();
    }

    ~PiecewisePolynomial() {}

    /**
     * @brief determination of a 3x3 matrix
     * @param m
     * @return double
     */
    double det_3x3(const double* m) {
        return m[0] * (m[4] * m[8] - m[7] * m[5]) - m[1] * (m[3] * m[8] - m[6] * m[5])
               + m[2] * (m[3] * m[7] - m[6] * m[4]);
    }

    /**
     * @brief calculate the inverse of a 3x3 matrix
     * @param input
     * @param output
     */
    void inverse_3x3(const double* input, double* output) {
        double det = det_3x3(input);
        output[0]  = (input[4] * input[8] - input[7] * input[5]) / det;
        output[3]  = (input[6] * input[5] - input[3] * input[8]) / det;
        output[6]  = (input[3] * input[7] - input[6] * input[4]) / det;

        output[1] = (input[7] * input[2] - input[1] * input[8]) / det;
        output[4] = (input[0] * input[8] - input[6] * input[2]) / det;
        output[7] = (input[1] * input[6] - input[0] * input[7]) / det;

        output[2] = (input[1] * input[5] - input[4] * input[2]) / det;
        output[5] = (input[3] * input[2] - input[0] * input[5]) / det;
        output[8] = (input[0] * input[4] - input[1] * input[3]) / det;
    }

    template <typename TV, typename T>
    void evaluate_vertices(const TV* function, const size_t* dofmap, const int4* cells, const double3* vertices,
                           TV* results, size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            const double3 local_points[4] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

            // Get local dofs from function and dofmap
            TV dof[num_local_dofs()];
            for (int i = 0; i < num_local_dofs(); i++) {
                dof[i] = function[dofmap[index * num_local_dofs() + i]];
            }

            // Evaluate at quadrature points
            TV local_results[4];
            TV dof_params[num_local_dofs()];
            transform_dofs<TV>(dof_params, dof);
            evaluate<TV, T>(dof_params, local_points, local_results, 4);

            // Evaluate at quadrature points
            // TV local_results[4];
            // evaluate<TV, T>(dof, local_points, local_results, 4);

            results[cells[index].x] = local_results[0];
            results[cells[index].y] = local_results[1];
            results[cells[index].z] = local_results[2];
            results[cells[index].w] = local_results[3];
        }
    }

    template <typename TV, typename T>
    void evaluate_grad_quadrature_points(const TV* function, const size_t* dofmap, const double* inv_Hs, TV* results,
                                                                 size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            TV* result = &(results[num_gauss * index * 3]);
            const double * inv_H  = &(inv_Hs[9 * index]);

            // Get local dofs from function and dofmap
            TV dof[num_local_dofs()];
            for (int i = 0; i < num_local_dofs(); i++) {
                dof[i] = function[dofmap[index * num_local_dofs() + i]];
            }

            // Evaluate at quadrature points
            TV dof_params[num_local_dofs()];
            transform_dofs<TV>(dof_params, dof);

            // 计算局部坐标系下的梯度
            evaluate_grad_local<TV, T>(dof_params, quadrature_points, inv_H, result, num_gauss);
        }
    }

    template <typename TV, typename T>
    void evaluate_div_quadrature_points(const TV* function, const size_t* dofmap, const double* inv_Hs, TV* results,
                                                                 size_t num_cells) {
        for (size_t index = 0; index < num_cells; index++) {
            TV* result = &(results[num_gauss * index]);
            const double * inv_H  = &(inv_Hs[9 * index]);

            // Get local dofs from function and dofmap
            TV dof[num_local_dofs()];
            for (int i = 0; i < num_local_dofs(); i++) {
                dof[i] = function[dofmap[index * num_local_dofs() + i]];
            }

            // Evaluate at quadrature points
            TV dof_params[num_local_dofs()];
            transform_dofs<TV>(dof_params, dof);

            // 计算局部坐标系下的梯度
            evaluate_div_local<TV, T>(dof_params, quadrature_points, inv_H, result, num_gauss);
        }
    }

    /**
     * @brief Calculate the volume of a tetrahedron
     * @param points
     * @return double
     */
    double tetrahedron_volume(const double3* points) {
        // Check that we get a tetrahedr
        // Get the coordinates of the four vertices
        const double* x0 = (double*)&(points[0]);
        const double* x1 = (double*)&(points[1]);
        const double* x2 = (double*)&(points[2]);
        const double* x3 = (double*)&(points[3]);

        // Formula for volume from http://mathworld.wolfram.com
        // I see this formula in /dolfin/mesh/TetrahedronCell.cpp which is a part of
        // fenics.
        const double v
            = (x0[0] * (x1[1] * x2[2] + x3[1] * x1[2] + x2[1] * x3[2] - x2[1] * x1[2] - x1[1] * x3[2] - x3[1] * x2[2])
               - x1[0] * (x0[1] * x2[2] + x3[1] * x0[2] + x2[1] * x3[2] - x2[1] * x0[2] - x0[1] * x3[2] - x3[1] * x2[2])
               + x2[0] * (x0[1] * x1[2] + x3[1] * x0[2] + x1[1] * x3[2] - x1[1] * x0[2] - x0[1] * x3[2] - x3[1] * x1[2])
               - x3[0]
                     * (x0[1] * x1[2] + x1[1] * x2[2] + x2[1] * x0[2] - x1[1] * x0[2] - x2[1] * x1[2] - x0[1] * x2[2]));

        return std::abs(v) / 6.0;
    }

    /**
     * @brief Get the transformation operator object
     * @param p Points of a cell, p0.x p0.y p0.z p1.x p1.y p1.z p2.x p2.y p2.z
     * @param H
     * @param inv_H
     * @param inv_Hb
     * @param b
     */
    void get_transformation_operator(const double* p, double* H, double* b, double* inv_H, double* inv_Hb);

    /**
     * @brief transform a point.
     *        It can be transformed from local cell to reference cell with H
     *        It also can be transformed from reference cell to local cell with
     * H_inv
     * @param point_out
     * @param point_in
     * @param A
     * @param b
     */
    void transform_a_point(double* point_out, const double* point_in, const double* A, const double* b);

    template <typename TV>
    void transform_dofs(TV* dofs_output, const TV* dofs_input);

    // 计算函数值(准备放弃第一个)
    template <typename TV, typename T>
    void evaluate(const TV* dofs, const double* points, TV* results, size_t num);

    template <typename TV, typename T>
    void evaluate(const TV* dofs, const double3* points, TV* results, size_t num);

    // NOTE: 计算函数参考点上的值。
    template <typename TV, typename T>
    void evaluate_points_all_cells(const TV* function, const size_t* dofmap, const double3* points, TV* results,
                                   size_t num_points, size_t num_cells);

    template <typename TV, typename T>
    void evaluate_quadrature_points_cpu(const TV* function, const size_t* dofmap, TV* results, size_t num_cells);

    template <typename TV, typename T>
    void evaluate_quadrature_points(const TV* function, const size_t* dofmap, TV* results, size_t num_cells);

    template <typename TV, typename T>
    void evaluate_surface_quadrature_points_cpu(const TV* function, const size_t* dofmap, TV* results,
                                                size_t num_cells);

    template <typename TV, typename T>
    void evaluate_surface_quadrature_points(const TV* function, const size_t* dofmap, TV* results, size_t num_cells);

    // input:
    // [f1,f2,f3,
    //  f1,f2,f3, ...]
    // output:
    // [f1x,f1y,f1z,f2x,f2y,f2z,f3x,f3y,f3z,
    //  f1x,f1y,f1z,f2x,f2y,f2z,f3x,f3y,f3z, ...]
    template <typename TV, typename T>
    void evaluate_grad_reference(const TV* dofs, const double* points, TV* results, size_t num);
    
    template <typename TV, typename T>
    void evaluate_grad_local(const TV* dofs, const double* points_ref, const double* inv_H, TV* results, size_t num);

    template <typename TV, typename T>
    void evaluate_div_local(const TV* dofs, const double* points_ref, const double* inv_H, TV* results, size_t num);
    
    // calculate first piola kirchhoff stress P(F)
    void calculate_PK_1(const double* F, double* PK_1, size_t num);

    /// evaluate the divergence of a matrix, such as first piola kirchhoff
    /// stress(PK_1). PK_1 is a matrix of 9*10 on local cell. temp variable is a
    /// matrix of 27*10 on reference cell. final results is a matrix of 3*10 on
    /// local cel.
    ///
    /// 	P = [P11, P12, P13,
    ///			 P21, P22, P23,
    ///			 P31, P32, P33].
    ///
    /// 	f1 = \\nabla_x P11 + \\nabla_y P12 + \\nabla_z P13.
    /// 	f2 = \\nabla_x P21 + \\nabla_y P22 + \\nabla_z P23.
    /// 	f3 = \\nabla_x P31 + \\nabla_y P32 + \\nabla_z P33.
    void evaluate_div_matrix_local(const double* dofs, const double* points_ref, const double* inv_H, double* results,
                                   size_t num);

    void get_local_quadrature_rules(const double* Hs, const double* bs, const double* volumes,
                                    double4* quadrature_rules, size_t num_cells);

    // NOTE : can be removed in the future.
    static const double quadratic_basis_dofs[10][10];
    static const double linear_basis_dofs[4][4];

    void print_basis_values_on_quadrature_points();
};

// NOTE : calculation of Gauss points.
// #include <dolfin/geometry/SimplexQuadrature.h>
// SimplexQuadrature gauss_quadrature(dim, order);
// ufc::cell ufc_cell;
// cell->get_cell_data(ufc_cell);
// auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);

// NOTE : the number of quadrature points should be larger than unknowns on
//        every cell.
//
// Explanation : http://talk.mapengfei.xyz/t/topic/137
template <int degree>
const double3 PiecewisePolynomial<degree>::tetrahedron_vertices[4]
    = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

// /// dimension : 3
// /// points on every direction : 1
// /// precision order : 1
// const double3 PiecewisePolynomial::points_1[1] = {{0.25,0.25,0.25}};
// const double PiecewisePolynomial::weights_1[1] = {0.1666666666666667};
// /// dimension : 3
// /// points on every direction : 2
// /// precision order : 3
// const double3 PiecewisePolynomial::points_2[8] = {
//     {0.1566826373368183,0.136054976802846, 0.1225148226554414},
//     {0.0813956670146703,0.0706797241593969,0.5441518440112253},
//     {0.0658386870600444,0.5659331650728009,0.1225148226554414},
//     {0.0342027932367664,0.2939988006316229,0.5441518440112253},
//     {0.5847475632048944,0.136054976802846, 0.1225148226554414},
//     {0.3037727648147076,0.0706797241593969,0.5441518440112253},
//     {0.2457133252117134,0.5659331650728009,0.1225148226554414},
//     {0.1276465621203854,0.2939988006316229,0.5441518440112253}
// };
// const double PiecewisePolynomial::weights_2[8] = {
//     0.0369798563588529,0.0160270405984766,
//     0.0211570064545241,0.0091694299214797,
//     0.0369798563588529,0.0160270405984766,
//     0.0211570064545241,0.0091694299214797
// };

/// dimension : 3
/// points on every direction : 3
/// precision order : 5
template <int degree>
const double3 PiecewisePolynomial<degree>::points_3[27]
    = {{0.0952198798417149, 0.0821215678634425, 0.0729940240731498},
       {0.0670742417520586, 0.0578476039361427, 0.3470037660383519},
       {0.0303014811742758, 0.0261332522867349, 0.7050022098884984},
       {0.0616960186091465, 0.3795782302805906, 0.0729940240731498},
       {0.0434595556538024, 0.2673803204118845, 0.3470037660383519},
       {0.0196333029354845, 0.1207918201339025, 0.7050022098884984},
       {0.0221843026408197, 0.7301650280476316, 0.0729940240731498},
       {0.0156269392579017, 0.514338662174092, 0.3470037660383519},
       {0.0070596311395548, 0.2323578005798647, 0.7050022098884984},
       {0.422442204031704, 0.0821215678634425, 0.0729940240731498},
       {0.2975743150127528, 0.0578476039361427, 0.3470037660383519},
       {0.1344322689123834, 0.0261332522867349, 0.7050022098884984},
       {0.2737138728231299, 0.3795782302805906, 0.0729940240731498},
       {0.1928079567748818, 0.2673803204118845, 0.3470037660383519},
       {0.0871029849887995, 0.1207918201339025, 0.7050022098884984},
       {0.0984204739396093, 0.7301650280476316, 0.0729940240731498},
       {0.0693287858937781, 0.514338662174092, 0.3470037660383519},
       {0.0313199947658185, 0.2323578005798647, 0.7050022098884984},
       {0.7496645282216929, 0.0821215678634425, 0.0729940240731498},
       {0.528074388273447, 0.0578476039361427, 0.3470037660383519},
       {0.238563056650491, 0.0261332522867349, 0.7050022098884984},
       {0.4857317270371133, 0.3795782302805906, 0.0729940240731498},
       {0.3421563578959613, 0.2673803204118845, 0.3470037660383519},
       {0.1545726670421146, 0.1207918201339025, 0.7050022098884984},
       {0.174656645238399, 0.7301650280476316, 0.0729940240731498},
       {0.1230306325296546, 0.514338662174092, 0.3470037660383519},
       {0.0555803583920821, 0.2323578005798647, 0.7050022098884984}};

template <int degree>
const double PiecewisePolynomial<degree>::weights_3[27]
    = {0.0087704749296511, 0.0081626507665467, 0.0016716811314837, 0.0100061425721761, 0.0093126823794705,
       0.0019072034149818, 0.0030478770905182, 0.0028366486956309, 0.0005809353158374, 0.0140327598874417,
       0.0130602412264747, 0.0026746898103739, 0.0160098281154818, 0.0149002918071527, 0.0030515254639709,
       0.0048766033448291, 0.0045386379130095, 0.0009294965053398, 0.0087704749296511, 0.0081626507665467,
       0.0016716811314837, 0.0100061425721761, 0.0093126823794705, 0.0019072034149818, 0.0030478770905182,
       0.0028366486956309, 0.0005809353158374};

template <int degree>
const double3 PiecewisePolynomial<degree>::quadratic_dof_coordinates[10]
    = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.5, 0.5},
       {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.5, 0.0}, {0.5, 0.0, 0.0}};

template <int degree>
const double3 PiecewisePolynomial<degree>::linear_dof_coordinates[4]
    = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

/// Can be removed in the future.
template <int degree>
const double PiecewisePolynomial<degree>::quadratic_basis_dofs[10][10]
    = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

/// Can be removed in the future.
template <int degree>
const double PiecewisePolynomial<degree>::linear_basis_dofs[4][4]
    = {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};

// const double PiecewisePolynomial::basis_values_1[10] = {
//     -0.125,
//     -0.125,
//     -0.125,
//     -0.125,
//     0.25,
//     0.25,
//     0.25,
//     0.25,
//     0.25,
//     0.25
// };

// const double PiecewisePolynomial::basis_values_2[80] = {
//     0.0991118621432293, -0.1192169795283644, -0.1249632488385189,
//     -0.0950592724780788, -0.1075837396511765, -0.0681451577971444,
//     -0.0571692216324634, -0.0318631311063724, -0.1075837396511765,
//     -0.0681451577971441, -0.0571692216324635, -0.0318631311063724,
//     0.0991118621432295, -0.1192169795283643, -0.1249632488385189,
//     -0.0950592724780786 , -0.0990330633772001, -0.0606884773449000,
//     0.0746275295858353, -0.1211282110859574, -0.0990330633772001,
//     -0.0606884773449000, 0.0746275295858353, -0.1211282110859574 ,
//     -0.0924950591148529, 0.0480506146704084, -0.0924950591148529,
//     0.0480506146704084, -0.0924950591148529, 0.0480506146704084,
//     -0.0924950591148529, 0.0480506146704084 , 0.0666750054175635,
//     0.1538420089421623, 0.2773408054149074, 0.6399199580031448,
//     0.0666750054175635, 0.1538420089421623, 0.2773408054149074,
//     0.6399199580031448 , 0.0767837821060285, 0.1771664092022260,
//     0.0322648602761138, 0.0744460520404844, 0.2865609760169966,
//     0.6611940405372456, 0.1204140978495675, 0.2778364486380045 ,
//     0.0852698103530782, 0.0230120931694681, 0.1490411862085144,
//     0.0402223207594428, 0.3182312645894506, 0.0858823008969635,
//     0.5562292793505097, 0.1501117446725730 , 0.2865609760169965,
//     0.6611940405372454, 0.1204140978495674, 0.2778364486380047,
//     0.0767837821060285, 0.1771664092022258, 0.0322648602761137,
//     0.0744460520404844 , 0.3182312645894505, 0.0858823008969635,
//     0.5562292793505095, 0.1501117446725730, 0.0852698103530781,
//     0.0230120931694680, 0.1490411862085141, 0.0402223207594428 ,
//     0.3664791615168828, 0.0989031472519347, 0.0647097709003877,
//     0.0174634758863504, 0.3664791615168823, 0.0989031472519346,
//     0.0647097709003875, 0.0174634758863504
// };

template <int degree>
const double PiecewisePolynomial<degree>::quadratic_basis_values_3[270]
    = {0.3743292815260137,  0.0296507308273032,  -0.1247383926536406, -0.0138611057361999, -0.1080144113987031,
       -0.1067872482490898, -0.1136467577865352, -0.0927575594483606, -0.0494020059140975, -0.0655273725373767,
       -0.1204733691021348, -0.0982881990625207, -0.1238753044714569, -0.1184581403834725, -0.0719291250008814,
       -0.0790472945586145, -0.0597158247867675, -0.0293581106215565, -0.0770862288075733, -0.0580763339388314,
       -0.0284651216515656, -0.0540832211847060, -0.0396820896985502, -0.0188623697671714, -0.0212000160735004,
       -0.0151385367967613, -0.0069599543559017, -0.0770862288075737, -0.0580763339388314, -0.0284651216515658,
       -0.0540832211847062, -0.0396820896985505, -0.0188623697671715, -0.0212000160735007, -0.0151385367967614,
       -0.0069599543559017, -0.0655273725373763, -0.1204733691021347, -0.0982881990625207, -0.1238753044714568,
       -0.1184581403834724, -0.0719291250008814, -0.0790472945586147, -0.0597158247867674, -0.0293581106215567,
       0.3743292815260140,  0.0296507308273035,  -0.1247383926536402, -0.0138611057361998, -0.1080144113987030,
       -0.1067872482490898, -0.1136467577865351, -0.0927575594483608, -0.0494020059140975, -0.0686336640467425,
       -0.0511549133738370, -0.0247673585365706, -0.0914189644747004, -0.1243958489247605, -0.0916104925113804,
       0.3361169083199657,  0.0147498566399774,  -0.1243775055992404, -0.0686336640467425, -0.0511549133738370,
       -0.0247673585365706, -0.0914189644747004, -0.1243958489247605, -0.0916104925113804, 0.3361169083199657,
       0.0147498566399774,  -0.1243775055992404, -0.0686336640467425, -0.0511549133738370, -0.0247673585365706,
       -0.0914189644747004, -0.1243958489247605, -0.0916104925113804, 0.3361169083199657,  0.0147498566399774,
       -0.1243775055992404, -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,  -0.0623377689723667,
       -0.1061805387487534, 0.2890540220068343,  -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,
       -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,  -0.0623377689723667, -0.1061805387487534,
       0.2890540220068343,  -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,  -0.0623377689723667,
       -0.1061805387487534, 0.2890540220068343,  -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,
       -0.0623377689723667, -0.1061805387487534, 0.2890540220068343,  0.0239775348061957,  0.0802933456885460,
       0.0736960024548870,  0.1108277699149801,  0.3711279125898606,  0.3406340005234211,  0.2131907345387237,
       0.7139098111741502,  0.6552510515745424,  0.0239775348061957,  0.0802933456885460,  0.0736960024548870,
       0.1108277699149801,  0.3711279125898606,  0.3406340005234211,  0.2131907345387237,  0.7139098111741502,
       0.6552510515745424,  0.0239775348061957,  0.0802933456885460,  0.0736960024548870,  0.1108277699149801,
       0.3711279125898606,  0.3406340005234211,  0.2131907345387237,  0.7139098111741502,  0.6552510515745424,
       0.0278019288056343,  0.0931000579685248,  0.0854504447630367,  0.0180137626702941,  0.0603225179288911,
       0.0553660878277077,  0.0064772860840401,  0.0216904270965778,  0.0199082222175352,  0.1233430256424186,
       0.4130376319428324,  0.3791001866542207,  0.0799179080880264,  0.2676203484921750,  0.2456311876199534,
       0.0287364257761546,  0.0962293992000303,  0.0883226620943930,  0.2188841224792030,  0.7329752059171400,
       0.6727499285454047,  0.1418220535057587,  0.4749181790554591,  0.4358962874121994,  0.0509955654682691,
       0.1707683713034829,  0.1567371019712505,  0.0312784232974809,  0.0155203366847607,  0.0031675050087564,
       0.0936738622360729,  0.0464809196626872,  0.0094861695872699,  0.0647928078398051,  0.0321501561271398,
       0.0065614414579683,  0.1387664645070872,  0.0688558444657068,  0.0140526095958620,  0.4155833097978012,
       0.2062122130417147,  0.0420853123835723,  0.2874527524583041,  0.1426338999470395,  0.0291097803918338,
       0.2462545057166935,  0.1221913522466528,  0.0249377141829676,  0.7374927573595296,  0.3659435064207422,
       0.0746844551798748,  0.5101126970768034,  0.2531176437669395,  0.0516581193256993,  0.2188841224792029,
       0.7329752059171397,  0.6727499285454044,  0.1418220535057586,  0.4749181790554589,  0.4358962874121997,
       0.0509955654682691,  0.1707683713034825,  0.1567371019712509,  0.1233430256424185,  0.4130376319428321,
       0.3791001866542205,  0.0799179080880264,  0.2676203484921749,  0.2456311876199540,  0.0287364257761546,
       0.0962293992000301,  0.0883226620943927,  0.0278019288056342,  0.0931000579685244,  0.0854504447630366,
       0.0180137626702941,  0.0603225179288909,  0.0553660878277078,  0.0064772860840401,  0.0216904270965775,
       0.0199082222175353,  0.2462545057166934,  0.1221913522466528,  0.0249377141829676,  0.7374927573595290,
       0.3659435064207421,  0.0746844551798749,  0.5101126970768033,  0.2531176437669391,  0.0516581193256993,
       0.1387664645070871,  0.0688558444657067,  0.0140526095958620,  0.4155833097978008,  0.2062122130417147,
       0.0420853123835724,  0.2874527524583042,  0.1426338999470393,  0.0291097803918338,  0.0312784232974809,
       0.0155203366847606,  0.0031675050087564,  0.0936738622360724,  0.0464809196626871,  0.0094861695872699,
       0.0647928078398050,  0.0321501561271396,  0.0065614414579683,  0.2855318651954619,  0.1416807567284945,
       0.0289152558798902,  0.1198708546813384,  0.0594798531131274,  0.0121390879903345,  0.0154985435047957,
       0.0076903688856085,  0.0015695073154094,  0.7138296629886549,  0.3542018918212358,  0.0722881396997255,
       0.2996771367033458,  0.1486996327828188,  0.0303477199758362,  0.0387463587619892,  0.0192259222140213,
       0.0039237682885236,  0.2855318651954617,  0.1416807567284939,  0.0289152558798901,  0.1198708546813381,
       0.0594798531131271,  0.0121390879903345,  0.0154985435047956,  0.0076903688856085,  0.0015695073154094};

template <int degree>
const double PiecewisePolynomial<degree>::linear_basis_values_3[27 * 4]
    = {0.7496645282216928, 0.5280743882734468, 0.2385630566504909, 0.4857317270371130, 0.3421563578959611,
       0.1545726670421147, 0.1746566452383989, 0.1230306325296544, 0.0555803583920822, 0.4224422040317037,
       0.2975743150127526, 0.1344322689123834, 0.2737138728231297, 0.1928079567748818, 0.0871029849887995,
       0.0984204739396093, 0.0693287858937780, 0.0313199947658185, 0.0952198798417148, 0.0670742417520584,
       0.0303014811742757, 0.0616960186091463, 0.0434595556538023, 0.0196333029354846, 0.0221843026408196,
       0.0156269392579015, 0.0070596311395549, 0.0952198798417149, 0.0670742417520586, 0.0303014811742758,
       0.0616960186091465, 0.0434595556538024, 0.0196333029354845, 0.0221843026408197, 0.0156269392579017,
       0.0070596311395548, 0.4224422040317040, 0.2975743150127528, 0.1344322689123834, 0.2737138728231299,
       0.1928079567748818, 0.0871029849887995, 0.0984204739396093, 0.0693287858937781, 0.0313199947658185,
       0.7496645282216929, 0.5280743882734470, 0.2385630566504910, 0.4857317270371133, 0.3421563578959613,
       0.1545726670421146, 0.1746566452383990, 0.1230306325296546, 0.0555803583920821, 0.0821215678634425,
       0.0578476039361427, 0.0261332522867349, 0.3795782302805906, 0.2673803204118845, 0.1207918201339025,
       0.7301650280476316, 0.5143386621740920, 0.2323578005798647, 0.0821215678634425, 0.0578476039361427,
       0.0261332522867349, 0.3795782302805906, 0.2673803204118845, 0.1207918201339025, 0.7301650280476316,
       0.5143386621740920, 0.2323578005798647, 0.0821215678634425, 0.0578476039361427, 0.0261332522867349,
       0.3795782302805906, 0.2673803204118845, 0.1207918201339025, 0.7301650280476316, 0.5143386621740920,
       0.2323578005798647, 0.0729940240731498, 0.3470037660383519, 0.7050022098884984, 0.0729940240731498,
       0.3470037660383519, 0.7050022098884984, 0.0729940240731498, 0.3470037660383519, 0.7050022098884984,
       0.0729940240731498, 0.3470037660383519, 0.7050022098884984, 0.0729940240731498, 0.3470037660383519,
       0.7050022098884984, 0.0729940240731498, 0.3470037660383519, 0.7050022098884984, 0.0729940240731498,
       0.3470037660383519, 0.7050022098884984, 0.0729940240731498, 0.3470037660383519, 0.7050022098884984,
       0.0729940240731498, 0.3470037660383519, 0.7050022098884984};

template <int degree>
void PiecewisePolynomial<degree>::get_transformation_operator(const double* p, double* H, double* b, double* inv_H,
                                                              double* inv_Hb) {
    H[0] = p[3] - p[0];
    H[1] = p[6] - p[0];
    H[2] = p[9] - p[0];

    H[3] = p[4] - p[1];
    H[4] = p[7] - p[1];
    H[5] = p[10] - p[1];

    H[6] = p[5] - p[2];
    H[7] = p[8] - p[2];
    H[8] = p[11] - p[2];

    inverse_3x3(H, inv_H);

    b[0] = p[0];
    b[1] = p[1];
    b[2] = p[2];

    inv_Hb[0] = -inv_H[0] * p[0] - inv_H[1] * p[1] - inv_H[2] * p[2];
    inv_Hb[1] = -inv_H[3] * p[0] - inv_H[4] * p[1] - inv_H[5] * p[2];
    inv_Hb[2] = -inv_H[6] * p[0] - inv_H[7] * p[1] - inv_H[8] * p[2];
}

template <int degree>
void PiecewisePolynomial<degree>::transform_a_point(double* point_out, const double* point_in, const double* A,
                                                    const double* b) {
    point_out[0] = A[0 * 3 + 0] * point_in[0] + A[0 * 3 + 1] * point_in[1] + A[0 * 3 + 2] * point_in[2];
    point_out[1] = A[1 * 3 + 0] * point_in[0] + A[1 * 3 + 1] * point_in[1] + A[1 * 3 + 2] * point_in[2];
    point_out[2] = A[2 * 3 + 0] * point_in[0] + A[2 * 3 + 1] * point_in[1] + A[2 * 3 + 2] * point_in[2];

    point_out[0] += b[0];
    point_out[1] += b[1];
    point_out[2] += b[2];
}

template <int degree>
constexpr int PiecewisePolynomial<degree>::num_local_dofs() {
    return -1;
}
template <>
constexpr int PiecewisePolynomial<1>::num_local_dofs() {
    return 4;
}
template <>
constexpr int PiecewisePolynomial<2>::num_local_dofs() {
    return 10;
}

template <int degree>
template <typename TV, typename T>
void PiecewisePolynomial<degree>::evaluate_quadrature_points_cpu(const TV* function, const size_t* dofmap, TV* results,
                                                                 size_t num_cells) {
    for (size_t index = 0; index < num_cells; index++) {
        TV* result = &(results[num_gauss * index]);

        // Get local dofs from function and dofmap
        TV dof[num_local_dofs()];
        for (int i = 0; i < num_local_dofs(); i++) {
            dof[i] = function[dofmap[index * num_local_dofs() + i]];
        }

        // Evaluate at quadrature points
        TV dof_params[num_local_dofs()];
        transform_dofs<TV>(dof_params, dof);
        evaluate<TV, T>(dof_params, quadrature_points, result, num_gauss);
    }
}

template <int degree>
template <typename TV, typename T>
void PiecewisePolynomial<degree>::evaluate_quadrature_points(const TV* function, const size_t* dofmap, TV* results,
                                                             size_t num_cells) {
    // if (use_gpu == true)
    // {
    //     gpu::evaluate_vector_function_for_quadrature_points(function, dofmap,
    //     (const double
    //     *)quadrature_points_device, results, num_gauss, num_cells);
    // }
    // else
    // {
    evaluate_quadrature_points_cpu<TV, T>(function, dofmap, results, num_cells);
    // }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<1>::evaluate_grad_reference(const TV* dofs, const double* points, TV* results,
                                                          size_t num) {
    for (size_t i = 0; i < num; i++) {
        // double x           = points[i].x;
        // double y           = points[i].y;
        // double z           = points[i].z;
        double x = points[3 * i];
        double y = points[3 * i + 1];
        double z = points[3 * i + 2];
        results[3 * i + 0] = dofs[1] - dofs[0]; // df1/dx
        results[3 * i + 1] = dofs[2] - dofs[0]; // df1/dy
        results[3 * i + 2] = dofs[3] - dofs[0]; // df1/dz
    }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<2>::evaluate_grad_reference(const TV* dofs, const double* points, TV* results,
                                                          size_t num) {
    for (size_t i = 0; i < num; i++) {
        // double x           = points[i].x;
        // double y           = points[i].y;
        // double z           = points[i].z;
        double x = points[3 * i];
        double y = points[3 * i + 1];
        double z = points[3 * i + 2];
        results[3 * i + 0] = dofs[1] + y * dofs[4] + z * dofs[5] + 2 * x * dofs[7]; // df1/dx
        results[3 * i + 1] = dofs[2] + x * dofs[4] + z * dofs[6] + 2 * y * dofs[8]; // df1/dy
        results[3 * i + 2] = dofs[3] + x * dofs[5] + y * dofs[6] + 2 * z * dofs[9]; // df1/dz
    }
}



template <int degree>
template <typename TV, typename T>
void PiecewisePolynomial<degree>::evaluate_grad_local(const TV* dofs, const double* points_ref, const double* inv_H,
                                                      TV* results, size_t num) {
    std::vector<TV> results_ref_v(num * 3);
    TV*             results_ref = results_ref_v.data();
    evaluate_grad_reference<TV, T>(dofs, points_ref, results_ref, num);

    for (size_t i = 0; i < num; i++) {
        results[3 * i + 0] = inv_H[0] * results_ref[3 * i + 0] + inv_H[3] * results_ref[3 * i + 1]
                             + inv_H[6] * results_ref[3 * i + 2]; // df1/dx
        results[3 * i + 1] = inv_H[1] * results_ref[3 * i + 0] + inv_H[4] * results_ref[3 * i + 1]
                             + inv_H[7] * results_ref[3 * i + 2]; // df1/dy
        results[3 * i + 2] = inv_H[2] * results_ref[3 * i + 0] + inv_H[5] * results_ref[3 * i + 1]
                             + inv_H[8] * results_ref[3 * i + 2]; // df1/dz
    }
}

template <int degree>
template <typename TV, typename T>
void PiecewisePolynomial<degree>::evaluate_div_local(const TV* dofs, const double* points_ref, const double* inv_H,
                                                      TV* results, size_t num) {
    std::vector<TV> results_ref_v(num * 3);
    TV*             results_ref = results_ref_v.data();
    evaluate_grad_reference<TV, T>(dofs, points_ref, results_ref, num);

    for (size_t i = 0; i < num; i++) {
        TV local_results[3];
        local_results[0] = inv_H[0] * results_ref[3 * i + 0] + inv_H[3] * results_ref[3 * i + 1]
                             + inv_H[6] * results_ref[3 * i + 2]; // df1/dx
        local_results[1] = inv_H[1] * results_ref[3 * i + 0] + inv_H[4] * results_ref[3 * i + 1]
                             + inv_H[7] * results_ref[3 * i + 2]; // df1/dy
        local_results[2] = inv_H[2] * results_ref[3 * i + 0] + inv_H[5] * results_ref[3 * i + 1]
                             + inv_H[8] * results_ref[3 * i + 2]; // df1/dz
        results[i] = local_results[0] + local_results[1] + local_results[2];
    }
}

// calculate first piola kirchhoff stress P(F)
template <int degree>
void PiecewisePolynomial<degree>::calculate_PK_1(const double* F, double* PK_1, size_t num) {
    // This an example, linear elastic solid
    double mu = 0.1;
    for (size_t j = 0; j < num; j++) {
        for (size_t i = 0; i < 9; i++) {
            PK_1[j * 9 + i] = mu * F[j * 9 + i];
        }
    }
}

/// evaluate the divergence of a matrix, such as first piola kirchhoff
/// stress(PK_1). PK_1 is a matrix of 9*10 on local cell. temp variable is a
/// matrix of 27*10 on reference cell. final results is a matrix of 3*10 on
/// local cel.
///
/// 	P = [P11, P12, P13,
///			 P21, P22, P23,
///			 P31, P32, P33].
///
/// 	f1 = \\nabla_x P11 + \\nabla_y P12 + \\nabla_z P13.
/// 	f2 = \\nabla_x P21 + \\nabla_y P22 + \\nabla_z P23.
/// 	f3 = \\nabla_x P31 + \\nabla_y P32 + \\nabla_z P33.
template <int degree>
void PiecewisePolynomial<degree>::evaluate_div_matrix_local(const double* dofs, const double* points_ref,
                                                            const double* inv_H, double* results, size_t num) {
    // TODO: assert num_gauss == num
    double results_ref[num_gauss * 27];
    evaluate_grad_matrix_reference(dofs, points_ref, results_ref, num);
    for (size_t i = 0; i < num; i++) {
        results[3 * i + 0] = inv_H[0] * results_ref[27 * i + 0] + inv_H[3] * results_ref[27 * i + 1]
                             + inv_H[6] * results_ref[27 * i + 2] + inv_H[1] * results_ref[27 * i + 3]
                             + inv_H[4] * results_ref[27 * i + 4] + inv_H[7] * results_ref[27 * i + 5]
                             + inv_H[2] * results_ref[27 * i + 6] + inv_H[5] * results_ref[27 * i + 7]
                             + inv_H[8] * results_ref[27 * i + 8];

        results[3 * i + 1] = inv_H[0] * results_ref[27 * i + 9] + inv_H[3] * results_ref[27 * i + 10]
                             + inv_H[6] * results_ref[27 * i + 11] + inv_H[1] * results_ref[27 * i + 12]
                             + inv_H[4] * results_ref[27 * i + 13] + inv_H[7] * results_ref[27 * i + 14]
                             + inv_H[2] * results_ref[27 * i + 15] + inv_H[5] * results_ref[27 * i + 16]
                             + inv_H[8] * results_ref[27 * i + 17];

        results[3 * i + 2] = inv_H[0] * results_ref[27 * i + 18] + inv_H[3] * results_ref[27 * i + 19]
                             + inv_H[6] * results_ref[27 * i + 20] + inv_H[1] * results_ref[27 * i + 21]
                             + inv_H[4] * results_ref[27 * i + 22] + inv_H[7] * results_ref[27 * i + 23]
                             + inv_H[2] * results_ref[27 * i + 24] + inv_H[5] * results_ref[27 * i + 25]
                             + inv_H[8] * results_ref[27 * i + 26];
    }
}

/// TODO : Coodrdinates are stored repeatedly.
/// TODO : there is no need to input gauss points on local cell.
/// TODO : I can output the local points
// template<int degree>
// void
// PiecewisePolynomial<degree>::calculate_body_force_from_displacement_kernel(
//     size_t num_cells,
//     size_t num_gauss,
//     const size_t* dofmap,
//     const double3 *function,
//     const double *inv_H_all,
//     double3 *results_all)
// {
//     for (size_t index = 0; index < num_cells; index++)
//     {
//         double *result = (double*)&(results_all[num_gauss * index]);
//         const double *points_ref = quadrature_points;
//         // Get local dofs from function and dofmap
//         double3 dof[10];
//         for (size_t i = 0; i < 10; i++)
//         {
//             dof[i] = function[dofmap[index*10+i]];
//         }
//         const double *local_dof = (double*)dof;
//         const double *inv_H = &(inv_H_all[9 * index]);
//         // for (size_t i = 0; i < 10; i++)
//         // {
//         //     printf("local_dof %ld : %.12e %.12e %.12e\n", i, local_dof[3*i
//         + 0], local_dof[3*i + 1], local_dof[3*i
//         + 2]);
//         // }
//         // step 3 : calculate deformation gradient
//         double parameters[30];
//         double deformation_dofs[90];
//         transform_dofs_vector(parameters, local_dof);
//         evaluate_grad_vector_local(parameters,
//         (double*)quadratic_dof_coordinates, inv_H, deformation_dofs, 10);
//         // for (size_t i = 0; i < 10; i++)
//         // {
//         //     printf("deformation_dofs %ld : %.12e %.12e %.12e\n", i,
//         deformation_dofs[9*i + 0], deformation_dofs[9*i + 1],
//         deformation_dofs[9*i + 2]);
//         //     printf("deformation_dofs %ld : %.12e %.12e %.12e\n", i,
//         deformation_dofs[9*i + 3], deformation_dofs[9*i + 4],
//         deformation_dofs[9*i + 5]);
//         //     printf("deformation_dofs %ld : %.12e %.12e %.12e\n", i,
//         deformation_dofs[9*i + 6], deformation_dofs[9*i + 7],
//         deformation_dofs[9*i + 8]);
//         // }
//         double temp_vector_1[90];
//         double PK_1_dofs[90];
//         // step 3 : calculate PK1 on gauss points
//         calculate_PK_1(deformation_dofs, PK_1_dofs, 10);
//         // for (size_t i = 0; i < 10; i++)
//         // {
//         //     printf("PK stress %ld : %.12e %.12e %.12e\n", i, PK_1_dofs[9*i
//         + 0], PK_1_dofs[9*i + 1], PK_1_dofs[9*i
//         + 2]);
//         //     printf("PK stress %ld : %.12e %.12e %.12e\n", i, PK_1_dofs[9*i
//         + 3], PK_1_dofs[9*i + 4], PK_1_dofs[9*i
//         + 5]);
//         //     printf("PK stress %ld : %.12e %.12e %.12e\n", i, PK_1_dofs[9*i
//         + 6], PK_1_dofs[9*i + 7], PK_1_dofs[9*i
//         + 8]);
//         // }
//         // step 4 : evaluate results
//         transform_dofs_matrix(temp_vector_1, PK_1_dofs);
//         evaluate_div_matrix_local(temp_vector_1, points_ref, inv_H, result,
//         num_gauss); for (size_t i = 0; i < 10; i++)
//         {
//             printf("force %ld : %.12e %.12e %.12e\n", i,  result[3*i + 0],
//             result[3*i + 1], result[3*i + 2]);
//         }
//     }
// }

template <int degree>
void PiecewisePolynomial<degree>::get_local_quadrature_rules(const double* Hs, const double* bs, const double* volumes,
                                                             double4* quadrature_rules, size_t num_cells) {
    //
    for (size_t index = 0; index < num_cells; index++) {
        const double* H = &(Hs[index * 9]);
        const double* b = &(bs[index * 3]);

        // TODO : Is for-loop can be optimized?
        for (size_t i = 0; i < num_gauss; i++) {
            // Transform reference quadrature point to local quadrature point
            double point[3];
            auto   quadrature_point = &(quadrature_points[3 * i]);
            transform_a_point(point, quadrature_point, H, b);

            // Write to quadrature rules
            auto quadrature_index                = index * num_gauss + i;
            quadrature_rules[quadrature_index].x = point[0];
            quadrature_rules[quadrature_index].y = point[1];
            quadrature_rules[quadrature_index].z = point[2];
            quadrature_rules[quadrature_index].w = quadrature_weights[i] * 6.0 * volumes[index];
        }
    }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<2>::evaluate(const TV* dofs, const double* points, TV* results, size_t num) {
    for (size_t i = 0; i < num; i++) {
        double x = points[3 * i];
        double y = points[3 * i + 1];
        double z = points[3 * i + 2];

        results[i] = dofs[0] + x * dofs[1] + y * dofs[2] + z * dofs[3] + x * y * dofs[4] + x * z * dofs[5]
                     + y * z * dofs[6] + x * x * dofs[7] + y * y * dofs[8] + z * z * dofs[9];
    }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<1>::evaluate(const TV* dofs, const double* points, TV* results, size_t num) {
    for (size_t i = 0; i < num; i++) {
        double x = points[3 * i];
        double y = points[3 * i + 1];
        double z = points[3 * i + 2];

        results[i] = dofs[0] * (1 - x - y - z) // base 1
                     + dofs[1] * x             // base 2
                     + dofs[2] * y             // base 3
                     + dofs[3] * z;            // base 4
    }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<2>::evaluate(const TV* dofs, const double3* points, TV* results, size_t num) {
    for (size_t i = 0; i < num; i++) {
        double x = points[i].x;
        double y = points[i].y;
        double z = points[i].z;

        results[i] = dofs[0] + x * dofs[1] + y * dofs[2] + z * dofs[3] + x * y * dofs[4] + x * z * dofs[5]
                     + y * z * dofs[6] + x * x * dofs[7] + y * y * dofs[8] + z * z * dofs[9];
    }
}

template <>
template <typename TV, typename T>
void PiecewisePolynomial<1>::evaluate(const TV* dofs, const double3* points, TV* results, size_t num) {
    for (size_t i = 0; i < num; i++) {
        double x = points[i].x;
        double y = points[i].y;
        double z = points[i].z;

        results[i] = dofs[0] * (1 - x - y - z) // base 1
                     + dofs[1] * x             // base 2
                     + dofs[2] * y             // base 3
                     + dofs[3] * z;            // base 4
    }
}

template <>
template <typename TV>
void PiecewisePolynomial<1>::transform_dofs(TV* dofs_output, const TV* dofs_input) {
    // do not do anything
    for (size_t i = 0; i < 4; i++) {
        dofs_output[i] = dofs_input[i];
    }
}

template <>
template <typename TV>
void PiecewisePolynomial<2>::transform_dofs(TV* dofs_output, const TV* dofs_input) {
    dofs_output[0] = 1.0 * dofs_input[0];

    dofs_output[1] = -3.0 * dofs_input[0] - 1.0 * dofs_input[1] + 4.0 * dofs_input[9];
    dofs_output[2] = -3.0 * dofs_input[0] - 1.0 * dofs_input[2] + 4.0 * dofs_input[8];
    dofs_output[3] = -3.0 * dofs_input[0] - 1.0 * dofs_input[3] + 4.0 * dofs_input[7];

    dofs_output[4] = 4.0 * dofs_input[0] + 4.0 * dofs_input[6] - 4.0 * dofs_input[8] - 4.0 * dofs_input[9];
    dofs_output[5] = 4.0 * dofs_input[0] + 4.0 * dofs_input[5] - 4.0 * dofs_input[7] - 4.0 * dofs_input[9];
    dofs_output[6] = 4.0 * dofs_input[0] + 4.0 * dofs_input[4] - 4.0 * dofs_input[7] - 4.0 * dofs_input[8];

    dofs_output[7] = 2.0 * dofs_input[0] + 2.0 * dofs_input[1] - 4.0 * dofs_input[9];
    dofs_output[8] = 2.0 * dofs_input[0] + 2.0 * dofs_input[2] - 4.0 * dofs_input[8];
    dofs_output[9] = 2.0 * dofs_input[0] + 2.0 * dofs_input[3] - 4.0 * dofs_input[7];
}

#endif