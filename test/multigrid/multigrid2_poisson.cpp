/// @date 2023-05-18
/// @file multigrid2_poisson.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief cell-centered multigrid method
///
///

#define CATCH_CONFIG_MAIN

#include <PhysicsSolver/multigrid/PoissonProblem1D2.h>
#include <PhysicsSolver/multigrid/set_boundary_types.h>
#include <catch.hpp>
#include <io/IBTimer.h>

using namespace pangu;

using BCVectorType = StdVector<char, char>;
using VectorType   = StdVector<double, double>;

double test_poisson_1d(int length) {
    // Define the mesh.
    int3    dim       = make_int3(length, 1, 1);
    int3    dim_ghost = make_int3(2, 0, 0);
    double3 dh        = {1.0 / (dim.x), 1.0 / (dim.y), 1.0 / (dim.z)};

    // Define the boundary type.
    BCVectorType boundary_type(dim + dim_ghost);
    boundary_type.data()[0]         = DIRICHLET;
    boundary_type.data()[dim.x + 1] = DIRICHLET;

    // Define the multigrid solver.
    PoissonProblem1D2<VectorType, BCVectorType> cpu_mgb(dim, dh);

    // Define variables
    StdVector<double, double> b(dim + dim_ghost);
    StdVector<double, double> x(dim + dim_ghost);
    StdVector<double, double> r(dim + dim_ghost);
    StdVector<double, double> x_exact(dim + dim_ghost);
    StdVector<double, double> x_old(dim + dim_ghost);

    // Calculate boundary conditions on every layer.
    cpu_mgb.compute_b(b);
    cpu_mgb.compute_exact(x_exact);

    // Solve
    cpu_mgb.smooth(x, b, boundary_type, dim, dh, 100000);
    cpu_mgb.compute_residuals(x, b, r, boundary_type, dim, dh);
    LOG_F(INFO, "the residual    :  %e!", std::sqrt(r.inner(r) / r.size()));

    // r = x-x_exact
    r.axpy(-1.0, x, x_exact);
    LOG_F(INFO, "the error       :  %e!", std::sqrt(r.inner(r) / r.size()));
    return std::sqrt(r.inner(r) / r.size());
}

// void print_2D_result(StdVector<double, double> x, int3 dim){
//     for (int j = 0; j < dim.y; j++){
//         for (int i = 0; i < dim.x; i++){
//             for (size_t n = 0; n < x.value_size(); n++)
//             {
//                 printf("%.1f", x.data()[i+dim.x*j]);
//             }
//             printf(", ");
//         }
//         printf("\n");
//     }
//     printf("end\n");

// }

// bool test_poisson_2d(int size){

//     // Define the mesh.
//     int3 dim = make_int3(size,size,1);
//     double3 dh = {1.0/(dim.x-1),1.0/(dim.y-1), 1.0/(dim.z-1)};

//     // Define the boundary type.
//     char BCtop    = NEUMANN;
//     char BCbottom = NEUMANN;
//     char BCleft   = NEUMANN;
//     char BCright  = NEUMANN;
//     BCVectorType boundary_type(dim);
//     for (int i = 0; i < dim.x; i++)
//     {
//         boundary_type.data()[i + 0*dim.x] = BCbottom;
//     }
//     for (int i = 0; i < dim.x; i++)
//     {
//         boundary_type.data()[i + (dim.y-1)*dim.x] = BCtop;
//     }
//     for (int i = 0; i < dim.y; i++)
//     {
//         boundary_type.data()[0 + i*dim.x] = BCleft;
//     }
//     for (int i = 0; i < dim.y; i++)
//     {
//         boundary_type.data()[dim.x-1 + i*dim.x] = BCright;
//     }

//     boundary_type.data()[0] = DIRICHLET;
//     boundary_type.data()[dim.x-1] = DIRICHLET;
//     boundary_type.data()[(dim.x-1)*dim.y] = DIRICHLET;
//     boundary_type.data()[(dim.x-1)*dim.y+dim.x-1] = DIRICHLET;

//     // Define the multigrid solver.
//     PoissonProblem2D<StdVector<double, double>, BCVectorType> cpu_mgb(dim,
//     dh);

//     // Calculate boundary conditions on every layer.
//     StdVector<double, double> x(dim);
//     StdVector<double, double> b(dim);
//     StdVector<double, double> r(dim);
//     StdVector<double, double> x_exact(dim);
//     StdVector<double, double> x_old(dim);

//     // Calculate boundary conditions on every layer.
//     cpu_mgb.calculate_bcs(boundary_type);
//     cpu_mgb.compute_b(b);
//     cpu_mgb.compute_exact(x_exact);
//     cpu_mgb.apply_dirichlet_bcs(x, x_exact, boundary_type);

//     // cpu_mgb.print_bcs_2D(0);
//     // print_2D_result(x, dim);
//     // print_2D_result(x_exact, dim);
//     // cpu_mgb.print_bcs_2D(1);
//     // cpu_mgb.print_bcs_2D(2);

//     // Solve
//     for (size_t i = 0; i < 100; i++)
//     {
//         // x_old = x;
//         LOG_F(INFO, "Step: %ld.", i);
//         cpu_mgb.iterate(x, b, 0);

//         // cpu_mgb.compute_residuals(x, b, r, boundary_type, dim, dh);
//         // LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!",
//         i, r.inner(r)/r.size());

//         // r.axpy(-1.0, x, x_old);
//         // LOG_F(INFO, "after %ld th iteration, the convergence is :  %e!",
//         i, r.inner(r)/r.size());

//         // r.axpy(-1.0, x, x_exact);
//         // LOG_F(INFO, "after %ld th iteration, the error is       :  %e!",
//         i, r.inner(r)/r.size());
//     }
//     return true;
// }

// bool test_poisson_3d(){

//     // int3 dim = make_int3(21,21,21);
//     // int3 dim_coarse = make_int3(11,11,11);
//     // double3 dh = make_double3(0.1,0.1,0.1);
//     // double3 dh_coarse = make_double3(0.2,0.2,0.2);

//     // BCVectorType fine(dim);
//     // BCVectorType coarse(dim_coarse);
//     // PoissonProblem3D<StdVector<double, double3>, BCVectorType>
//     cpu_mgb(dim, dh);

//     // StdVector<double, double3> x(dim);
//     // init_boundary_conditions(x,fine,dim);

//     // cpu_mgb.restrict_bcs(fine, coarse, dim, dim_coarse);
//     // cpu_mgb.print_bcs(fine, dim);
//     // cpu_mgb.print_bcs(coarse, dim_coarse);

//     // // Define the boundary type.
//     // char BCtop    = NEUMANN;
//     // char BCbottom = NEUMANN;
//     // char BCleft   = DIRICHLET;
//     // char BCright  = DIRICHLET;
//     // char BCfront  = NEUMANN;
//     // char BCback   = NEUMANN;
//     // double3 velocity_top    = make_double3(0,0,0);
//     // double3 velocity_bottom = make_double3(0,0,0);
//     // double3 velocity_left   = make_double3(0,0,0);
//     // double3 velocity_right  = make_double3(0,0,0);
//     // double3 velocity_front  = make_double3(0,0,0);
//     // double3 velocity_back   = make_double3(0,0,0);
//     // double pressure_top    = 0.0;
//     // double pressure_bottom = 0.0;
//     // double pressure_left   = 0.0;
//     // double pressure_right  = 0.0;
//     // double pressure_front  = 0.0;
//     // double pressure_back   = 0.0;

//     // Define the mesh.
//     int size   = 257;
//     int3 dim   = make_int3(size,size,size);
//     double3 dh = {1.0/(dim.x-1),1.0/(dim.y-1), 1.0/(dim.z-1)};

//     // Define the multigrid solver.
//     PoissonProblem3D<StdVector<double, double3>, BCVectorType> cpu_mgb(dim,
//     dh);

//     // Define variables
//     StdVector<double, double3> x(dim);
//     StdVector<double, double3> b(dim);
//     StdVector<double, double3> r(dim);
//     StdVector<double, double3> x_exact(dim);
//     StdVector<double, double3> x_old(dim);

//     // Define the boundary type.
//     BCVectorType boundary_type(dim);
//     init_boundary_conditions(x,boundary_type,dim);
//     // cpu_mgb.print_bcs(boundary_type, dim);

//     // Calculate boundary conditions on every layer.
//     cpu_mgb.compute_b(b);
//     cpu_mgb.compute_exact(x_exact);
//     cpu_mgb.calculate_bcs(boundary_type);
//     cpu_mgb.apply_dirichlet_bcs(x,x_exact);

//     // Solve
//     for (size_t i = 0; i < 10; i++)
//     {
//         x_old = x;
//         cpu_mgb.iterate(x, b, 0);

//         cpu_mgb.compute_residuals(x, b, r, boundary_type, dim, dh);
//         LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", i,
//         r.inner(r)/r.size());

//         r.axpy(-1.0, x, x_old);
//         LOG_F(INFO, "after %ld th iteration, the convergence is :  %e!", i,
//         r.inner(r)/r.size());

//         r.axpy(-1.0, x, x_exact);
//         LOG_F(INFO, "after %ld th iteration, the error is       :  %e!", i,
//         r.inner(r)/r.size());
//     }
//     return true;
// }

TEST_CASE("test poisson problems", "[long]") {
    double err = 1;
    for (int i = 64; i < 1 << 10; i *= 2) {
        double err_new = test_poisson_1d(i);
        LOG_F(INFO, "收敛阶 :  %e!", std::log10(err / err_new) / std::log10(2));
        err = err_new;
    }
    // REQUIRE(test_poisson_2d(513));
    // REQUIRE(test_poisson_3d());
}