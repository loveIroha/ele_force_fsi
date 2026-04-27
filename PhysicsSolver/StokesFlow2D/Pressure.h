#include <MultigridSolver2D/MultigridSolver.h>

#include "header.h"

#ifndef __PRESSURE_H__
#define __PRESSURE_H__

void gauss_seidel(const std::vector<std::vector<double>>& pb, std::vector<std::vector<double>>& phi,
                  const std::vector<std::vector<int>>&    boundary_type,
                  const std::vector<std::vector<double>>& boundary_values, double width, double height,
                  size_t max_iters, size_t Nx, size_t Ny) {
    double dx = width / Nx;
    double dy = height / Ny;

    auto residual = make_vector_2D<double>(Nx + 2, Ny + 2);

    for (size_t iter = 0; iter < max_iters; iter++) {
        // 处理边界条件
        for (size_t i = 1; i < Nx + 1; i++) {
            // 下边界
            if (boundary_type[i][0] == DIRICHLET) { phi[i][0] = 2 * boundary_values[i][0] - phi[i][1]; }
            if (boundary_type[i][0] == NEUMANN) { phi[i][0] = phi[i][1] - dy * boundary_values[i][0]; }
            // 上边界
            if (boundary_type[i][Ny + 1] == DIRICHLET) { phi[i][Ny + 1] = 2 * boundary_values[i][Ny + 1] - phi[i][Ny]; }
            if (boundary_type[i][Ny + 1] == NEUMANN) { phi[i][Ny + 1] = phi[i][Ny] + dy * boundary_values[i][Ny + 1]; }
        }
        for (size_t j = 1; j < Ny + 1; j++) {
            // 左边界
            if (boundary_type[0][j] == DIRICHLET) { phi[0][j] = 2 * boundary_values[0][j] - phi[1][j]; }
            if (boundary_type[0][j] == NEUMANN) { phi[0][j] = phi[1][j] - dx * boundary_values[0][j]; }
            // 右边界
            if (boundary_type[Nx + 1][j] == DIRICHLET) { phi[Nx + 1][j] = 2 * boundary_values[Nx + 1][j] - phi[Nx][j]; }
            if (boundary_type[Nx + 1][j] == NEUMANN) { phi[Nx + 1][j] = phi[Nx][j] + dx * boundary_values[Nx + 1][j]; }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double a  = 2.0 / dx / dx + 2.0 / dy / dy;
                double b  = ((phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy
                            - pb[i][j]);
                phi[i][j] = b / a;
                // printf(" b/a : %lf, phi[i][j]: %lf.\n", b/a, phi[i][j]);
            }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double a       = 2.0 / dx / dx + 2.0 / dy / dy;
                double b       = ((phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy
                            - pb[i][j]);
                residual[i][j] = phi[i][j] * a - b;
                // printf("residual[i][j] : %lf.\n", residual[i][j]);
            }
        }

        // 计算L2误差和L1误差
        double residual_norm_L2 = 0.0;
        double residual_norm_L1 = 0.0;

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                residual_norm_L1 = std::max(residual_norm_L1, std::abs(residual[i][j]));
                residual_norm_L2 += residual[i][j] * residual[i][j];
            }
        }
        residual_norm_L2 = std::sqrt(residual_norm_L2 * dx * dy);
        if (iter % 10000 == 0)
            printf("iter: %08zu, Error L1 : %.16e, L2 : %.16e.\n", iter, residual_norm_L1, residual_norm_L2);

        // 跳出循环
        if (residual_norm_L2 < EPSILON) break;
    }
}
int poisson(const std::vector<std::vector<double>>& pb, std::vector<std::vector<double>>& phi,
            const std::vector<std::vector<int>>& boundary_type, const std::vector<std::vector<double>>& boundary_values,
            double width, double height, double T, double rho, double mu, size_t max_iters, size_t Nx, size_t Ny,
            size_t Nt) {
    // double dx = width / Nx;
    // double dy = height / Ny;
    // double dt = T / Nt;

    // printf("Nx    : %d     , Ny     : %d     , Nt : %d     , max_iters :
    // %d.\n", Nx, Ny, Nt, max_iters); printf("width : %f, height : %f, T  : %f,
    // rho       : %f.\n", width, height, T, rho); printf("dx    : %f, dy     :
    // %f, dt : %f.\n", dx, dy, dt);

    // auto residual = make_vector_2D<double>(Nx + 2, Ny + 2);

    // gauss_seidel(pb, phi, boundary_type, boundary_values, width, height,
    // max_iters, Nx, Ny); IO::write_vector_2D(phi, "p_gauss_seidel.txt");

    auto u_prime = mg::generate_u_prime(boundary_type, boundary_values, Nx, Ny, width, height);
    auto b_prime = mg::compute_residual(u_prime, pb, Nx, Ny, width, height);

    double tol_relative    = 1e-10;
    double tol_descending  = 1e-10;
    int    num_presmooth   = 2;
    int    num_aftersmooth = 2;
    int    num_exactsmooth = 100;
    int    max_iteration   = 50;
    int    N_coarsest      = 3;

    mg::MultigridSolver2D solver(Nx, Ny, width, height, boundary_type, "cell-centered", tol_relative, tol_descending,
                                 num_presmooth, num_aftersmooth, num_exactsmooth, max_iteration, N_coarsest);
    phi = solver.vcycle(phi, b_prime);
    IO::write_vector_2D(phi, "p_multigrid.txt");

    return 0;
}

int poisson(const std::vector<std::vector<double>>& u, // (Nx+1)*(Ny+2)
            const std::vector<std::vector<double>>& v, // (Nx+2)*(Ny+1)
            std::vector<std::vector<double>>& phi, const std::vector<std::vector<double>>& s,
            const std::vector<std::vector<int>>& boundary_type, const std::vector<std::vector<double>>& boundary_values,
            double width, double height, double T, double rho, double mu, size_t max_iters, size_t Nx, size_t Ny,
            size_t Nt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    // 设置边界类型
    auto pb = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            pb[i + 1][j + 1]
                = rho / dt
                  * ((u[i + 1][j + 1] - u[i][j + 1]) / dx + (v[i + 1][j + 1] - v[i + 1][j]) / dy - s[i + 1][j + 1]);
        }
    }

    poisson(pb, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters, Nx, Ny, Nt);
    return 0;
}

int poisson(const std::vector<std::vector<double>>& u, // (Nx+1)*(Ny+2)
            const std::vector<std::vector<double>>& v, // (Nx+2)*(Ny+1)
            std::vector<std::vector<double>>& phi, const std::vector<std::vector<int>>& boundary_type,
            const std::vector<std::vector<double>>& boundary_values, double width, double height, double T, double rho,
            double mu, size_t max_iters, size_t Nx, size_t Ny, size_t Nt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    // 设置边界类型
    auto pb = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            pb[i + 1][j + 1] = rho / dt * ((u[i + 1][j + 1] - u[i][j + 1]) / dx + (v[i + 1][j + 1] - v[i + 1][j]) / dy);
        }
    }

    poisson(pb, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters, Nx, Ny, Nt);
    return 0;
}

#endif