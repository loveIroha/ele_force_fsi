/// @date 2023-08-12
/// @file multigrid.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///         2023/8/29 加入NPUHEART项目
///

#include <AlgebraSolver/algebra.h>
#include <MultigridSolver2D/MultigridSolver.h>
#include <io/vector_io.h>
using namespace mg;

int main() {
    int    Nx     = 256;
    int    Ny     = 256;
    double width  = 1.0;
    double height = 1.0;
    double dx     = width / Nx;
    double dy     = height / Ny;

    auto boundary_type   = IO::read_vector_2D<int>("boundary_type.txt");
    auto boundary_values = IO::read_vector_2D<double>("boundary_values.txt");
    auto u_exact         = IO::read_vector_2D<double>("u_exact.txt");
    auto b_original      = IO::read_vector_2D<double>("b_original.txt");

    auto u_prime = generate_u_prime(boundary_type, boundary_values, Nx, Ny, width, height);
    auto b_prime = compute_residual(u_prime, b_original, Nx, Ny, width, height);
    IO::write_vector_2D(b_prime, "b_prime_generated.txt");
    IO::write_vector_2D(u_prime, "u_prime_generated.txt");

    MultigridSolver2D solver(Nx, Ny, width, height, boundary_type);
    auto              x = IO::make_vector_2D<double>(Nx + 2, Ny + 2);
    x                   = solver.vcycle(x, b_prime);
    algebra::axpy(1.0, u_prime, x);
    algebra::zero_boundary(x);
    algebra::zero_boundary(u_exact);

    double error_norm2 = algebra::distance(x, u_exact) * std::sqrt(dx * dy);

    IO::write_vector_2D(x, "result.txt");
    printf("error_norm2 = %.12e\n", error_norm2);
    return 0;
}