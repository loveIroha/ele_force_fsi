/**
 * @file test_multigrid_stokes_cpu.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-03
 * @updated 2023-04-03
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

// NOTE:
// 从残差来看，网格大小为65，129时，小数点后12位是准确的，小数点后13位开始出现误差；
#include <PhysicsSolver/StokesFlow/PressureSolver.h>
#include <PhysicsSolver/StokesFlow/PressureSolverGPU.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <PhysicsSolver/StokesFlow/TentitiveVelocity.h>
#include <PhysicsSolver/StokesFlow/TentitiveVelocityGPU.h>
#include <PhysicsSolver/multigrid/set_boundary_types.h>
#include <catch.hpp>
#include <loguru/IBTimer.h>

using namespace pangu;

// 定义数据类型
using PressureType         = StdVector<double, double>;
using VelocityType         = StdVector<double, double3>;
using PressureBoundaryType = StdVector<char, char>;
using VelocityBoundaryType = StdVector<char, char>;

// CUDA数据类型
using GPUPressureType         = GpuVector<double, double>;
using GPUVelocityType         = GpuVector<double, double3>;
using GPUPressureBoundaryType = GpuVector<char, char>;
using GPUVelocityBoundaryType = GpuVector<char, char>;

using FluidSolver = ProjectionScheme<PressureType, VelocityType, PressureBoundaryType, VelocityBoundaryType>;
using GPUFluidSolver
    = ProjectionSchemeGPU<GPUPressureType, GPUVelocityType, GPUPressureBoundaryType, GPUVelocityBoundaryType>;

bool test_stokes_3d_gpu() {
    int     n   = 129;
    double  dt  = 1.0;
    double  mu  = 0.1;
    double  rho = 1.0;
    int3    dim = make_int3(n, n, n);
    double3 dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    VelocityType         f(dim);
    VelocityType         u0(dim);
    PressureType         p0(dim);
    PressureBoundaryType pbcs(dim);
    VelocityBoundaryType vbcs(dim);

    GPUVelocityType         gpu_f(dim);
    GPUVelocityType         gpu_u0(dim);
    GPUPressureType         gpu_p0(dim);
    GPUPressureBoundaryType gpu_pbcs(dim);
    GPUVelocityBoundaryType gpu_vbcs(dim);

    init_boundary_conditions(u0, vbcs, dim, DIRICHLET, double3{0, 0, 1}, DIRICHLET, double3{0, 0, 0}, DIRICHLET,
                             double3{0, 0, 0}, DIRICHLET, double3{0, 0, 0}, NEUMANN, double3{0, 0, 0}, NEUMANN,
                             double3{0, 0, 0});

    init_boundary_conditions(p0, pbcs, dim, NEUMANN, 0.0, NEUMANN, 0.0, NEUMANN, 0.0, NEUMANN, 0.0, DIRICHLET, 0.0,
                             DIRICHLET, 0.0);

    // 将边界类型和边界值复制到GPU变量中
    gpu_u0   = u0;
    gpu_p0   = p0;
    gpu_pbcs = pbcs;
    gpu_vbcs = vbcs;

    FluidSolver    projection_scheme(dim, dh, dt, mu, rho);
    GPUFluidSolver gpu_projection_scheme(dim, dh, dt, mu, rho);

    // testing running.
    for (size_t i = 0; i < 2; i++) {
        // 设置时间步长
        projection_scheme._dt = dt;
        projection_scheme._t += dt;
        gpu_projection_scheme._dt = dt;
        gpu_projection_scheme._t += dt;

        // 处理边界条件
        projection_scheme.vbcs     = vbcs;
        projection_scheme.pbcs     = pbcs;
        gpu_projection_scheme.vbcs = vbcs;
        gpu_projection_scheme.pbcs = pbcs;

        // TODO: 复制多重网格求解器的边界类型。
        projection_scheme.tv->calculate_bcs(vbcs);
        projection_scheme.ps->calculate_bcs(pbcs);

        for (int i = 0; i < projection_scheme.tv->num_levels; i++) {
            gpu_projection_scheme.tv->multi_bc[i] = projection_scheme.tv->multi_bc[i];
            gpu_projection_scheme.ps->multi_bc[i] = projection_scheme.ps->multi_bc[i];
        }

        // 推进时间步
        gpu_u0 = gpu_projection_scheme.solveOneStep(gpu_f, gpu_u0);
        u0     = projection_scheme.solveOneStep(f, u0);

        // 保存结果
        projection_scheme.record();
    }

    return true;
}

// bool test_stokes_3d(){

//     int n = 65;
//     double dt = 1;
//     double mu = 0.1;
//     double rho = 1.0;
//     int3 dim = make_int3(n, n, n);
//     double3 dh = {1.0/(dim.x-1), 1.0/(dim.y-1), 1.0/(dim.z-1)};
//     VelocityType f(dim);
//     VelocityType u0(dim);
//     PressureType p0(dim);

//     PressureBoundaryType pbcs(dim);
//     VelocityBoundaryType vbcs(dim);

//     init_boundary_conditions(
//         u0, vbcs, dim,
//         DIRICHLET, double3{0,0,1},
//         DIRICHLET, double3{0,0,0},
//         DIRICHLET, double3{0,0,0},
//         DIRICHLET, double3{0,0,0},
//         NEUMANN,   double3{0,0,0},
//         NEUMANN,   double3{0,0,0});

//     init_boundary_conditions(
//         p0, pbcs, dim,
//         NEUMANN,   0.0,
//         NEUMANN,   0.0,
//         NEUMANN,   0.0,
//         NEUMANN,   0.0,
//         DIRICHLET, 0.0,
//         DIRICHLET, 0.0);

//     FluidSolver projection_scheme(dim, dh, dt, mu, rho);

//     // testing running.
//     for (size_t i = 0; i < 2; i++){

//         // 设置时间步长
//         projection_scheme._dt = dt;
//         projection_scheme._t += dt;

//         // 处理边界条件
//         projection_scheme.vbcs=vbcs;
//         projection_scheme.pbcs=pbcs;
//         projection_scheme.tv->calculate_bcs(vbcs);
//         projection_scheme.ps->calculate_bcs(pbcs);

//         // 推进时间步
//         u0 = projection_scheme.solveOneStep(f, u0);

//         // 保存结果
//         projection_scheme.record();
//     }

//     return true;
// }

TEST_CASE("test stokes problems", "[long]") {
    // REQUIRE(test_poisson_1d(8193));
    // REQUIRE(test_poisson_2d(513));
    // REQUIRE(test_stokes_3d());
    REQUIRE(test_stokes_3d_gpu());
}