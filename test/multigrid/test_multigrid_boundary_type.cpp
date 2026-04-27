/// @date 2022-01-13
/// @file test_multigrid_boundary_type.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.2
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief test restriction and prolongation functions
///
/// updated at 2023-03-20

#include <PhysicsSolver/multigrid/MultigridBase.h>
#include <PhysicsSolver/multigrid/PoissonProblem1D.h>
#include <PhysicsSolver/multigrid/PoissonProblem2D.h>
#include <PhysicsSolver/multigrid/PoissonProblem3D.h>
#include <catch.hpp>

using namespace pangu;

using BCVectorType = StdVector<char, char3>;

// 参考论文 : A parallel multigrid Poisson solver for fluids simulation on large
// grids 边界类型分为三种，Neumann，Dirichlet，Inner
// 边界类型做restriction结果如下：
// 1 0 0 0 0 1 1 1 0 0 0
// 1   0   0   1   0   0
// 1 0 0 0 1 1 1 0 0 0 0
// 1   0   0   0   0   0
// 初始化时对边界进行一次restrict计算，结果保存在多重网格对象中。
// NOTE: 论文中默认网格最外层包裹着一层Neumann单元，smooth只遍历interior。
// NOTE: 速度Neumann边界条件不会影响解的唯一性，也不会影响求解器的收敛性，但是
// NOTE: 压强Neumann边界条件对解的影响较为重大。
// Neumann边界条件对残差算子和smooth算子的影响。
// Neumann边界条件对restriction和prolongation的影响。

int test_multigrid_boundary_type_3D() {
    int3    dim        = make_int3(21, 21, 21);
    int3    dim_coarse = make_int3(11, 11, 11);
    double3 dh         = make_double3(0.1, 0.1, 0.1);
    double3 dh_coarse  = make_double3(0.2, 0.2, 0.2);

    BCVectorType fine(dim);
    BCVectorType coarse(dim_coarse);

    PoissonProblem3D<StdVector<double, double3>, BCVectorType> poisson_solver(dim, dh);

    // test (x = 0, y)
    for (size_t i = 0; i < 21; i++)
        for (size_t j = 0; j < 21; j++) {
            fine.data()[i + j * 21]                = make_char3(1, 1, 2);
            fine.data()[i + j * 21 + 20 * 21 * 21] = make_char3(1, 1, 2);

            fine.data()[i + 0 * 21 + j * 21 * 21]  = make_char3(1, 1, 2);
            fine.data()[i + 20 * 21 + j * 21 * 21] = make_char3(1, 1, 2);

            fine.data()[0 + i * 21 + j * 21 * 21]  = make_char3(1, 1, 2);
            fine.data()[20 + i * 21 + j * 21 * 21] = make_char3(1, 1, 2);
        }

    // 测试边界条件的 restriction and prolongation
    fine.data()[1 + 3 * 21] = make_char3(0, 0, 0);

    fine.data()[19 + 17 * 21 + 20 * 21 * 21] = make_char3(0, 0, 0);

    poisson_solver.restrict_bcs(fine, coarse, dim, dim_coarse);

    poisson_solver.print_bcs(fine, dim);
    poisson_solver.print_bcs(coarse, dim_coarse);

    return 1;
}

// int3 dim = make_int3(21,1,1);
// int3 dim_coarse = make_int3(11,1,1);
// double3 dh = make_double3(0.1,0,0);
// double3 dh_coarse = make_double3(0.2,0,0);

// PoissonProblem1D<StdVector<double, double3>> poisson_solver(dim,dh);

// BCVectorType fine(dim);
// BCVectorType coarse(dim_coarse);

// fine.data()[0] = make_char3(1,1,1);
// fine.data()[3] = make_char3(1,1,0);
// fine.data()[4] = make_char3(1,1,2);
// fine.data()[5] = make_char3(1,1,2);
// fine.data()[6] = make_char3(1,1,2);
// fine.data()[7] = make_char3(1,1,2);

// Results for boundary restriction:

// poisson_solver.restrict_bcs(fine, coarse, dim, dim_coarse);

// printf("%d \n", poisson_solver.find_interior_cell(4, 0, dim, fine));
// printf("%d \n", poisson_solver.find_interior_cell(4, 1, dim, fine));
// printf("%d \n", poisson_solver.find_interior_cell(4, 2, dim, fine));
// printf("%d \n", poisson_solver.find_interior_cell(7, 2, dim, fine));

// print_bcs(fine, dim);
// print_bcs(coarse, dim_coarse);

// int3 dim = make_int3(21,21,1);
// int3 dim_coarse = make_int3(11,11,1);
// double3 dh = make_double3(0.1,0.1,0);
// double3 dh_coarse = make_double3(0.2,0.2,0);

// BCVectorType fine(dim);
// BCVectorType coarse(dim_coarse);

// PoissonProblem2D<StdVector<double, double3>> poisson_solver(dim,dh);

// // test (x = 0, y)
// for (size_t i = 0; i < 21; i++)
// {
//     fine.data()[i] = make_char3(1,1,2);
//     fine.data()[i+20*21] = make_char3(1,1,2);
//     fine.data()[i*21] = make_char3(1,1,2);
//     fine.data()[20+i*21] = make_char3(1,1,2);
// }
// fine.data()[5] = make_char3(1,1,0);

// fine.data()[3+3*21] = make_char3(1,1,2);
// fine.data()[4+3*21] = make_char3(1,1,2);
// fine.data()[5+3*21] = make_char3(1,1,2);

// fine.data()[3+4*21] = make_char3(1,1,2);
// fine.data()[4+4*21] = make_char3(1,1,2);
// fine.data()[5+4*21] = make_char3(1,1,2);

// fine.data()[3+5*21] = make_char3(1,1,2);
// fine.data()[4+5*21] = make_char3(1,1,2);
// fine.data()[5+5*21] = make_char3(1,1,2);

// poisson_solver.restrict_bcs(fine, coarse, dim, dim_coarse);

// print_bcs(fine, dim);
// print_bcs(coarse, dim_coarse);

TEST_CASE("test_email", "[email]") { REQUIRE(test_multigrid_boundary_type_3D()); }
