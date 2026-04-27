/// @date 2023-08-12
/// @file multigrid.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include "MultigridSolver.h"
#include "algebra.h"
using namespace mg;

void zero_boundary(std::vector<std::vector<double>>& x) {
    for (int i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < x[i].size(); j++) {
            if (i == 0 || i == x.size() - 1 || j == 0 || j == x[i].size() - 1) { x[i][j] = 0.0; }
        }
    }
}

int main() {
    int    Nx     = 256;
    int    Ny     = 256;
    double width  = 1.0;
    double height = 1.0;
    double dx     = width / Nx;
    double dy     = height / Ny;

    auto boundary_type = read_vector_2D<int>("boundary_type.txt");
    auto u_exact       = read_vector_2D<double>("u_exact.txt");
    auto b             = read_vector_2D<double>("b.txt");
    auto phi_prime     = read_vector_2D<double>("phi_prime.txt");

    MultigridSolver2D solver(Nx, Ny, width, height, boundary_type);
    auto              x = make_vector_2D<double>(Nx + 2, Ny + 2);
    x                   = solver.vcycle(x, b);
    algebra::axpy(1.0, phi_prime, x);
    zero_boundary(x);
    zero_boundary(u_exact);

    double error_norm2 = algebra::distance(x, u_exact) * std::sqrt(dx * dy);

    write_vector_2D(x, "result.txt");
    printf("error_norm2 = %.12e\n", error_norm2);
    return 0;
}