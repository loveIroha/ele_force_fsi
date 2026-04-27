#include "header.h"
#ifndef __TENTITIVE_VELOCITY_U_H__
#define __TENTITIVE_VELOCITY_U_H__

int tentitive_velocity_u(const std::vector<std::vector<double>>& ub, //(Nx+1)*(Ny+2)
                         std::vector<std::vector<double>>& phi, const std::vector<std::vector<int>>& boundary_type,
                         const std::vector<std::vector<double>>& boundary_values, double width, double height, double T,
                         double rho, double mu, size_t max_iters, size_t Nx, size_t Ny, size_t Nt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    // printf("Nx    : %05d     , Ny     : %05d     , Nt : %05d     , max_iters :
    // %05d.\n", Nx, Ny, Nt, max_iters); printf("width : %f, height : %f, T  : %f,
    // rho       : %f.\n", width, height, T, rho); printf("dx    : %f, dy     :
    // %f, dt : %f.\n", dx, dy, dt);

    auto residual = make_vector_2D<double>(Nx + 1, Ny + 2);

    for (size_t iter = 0; iter < max_iters; iter++) {
        for (size_t i = 0; i < Nx + 1; i++) {
            // 下边界
            if (boundary_type[i][0] == DIRICHLET) { phi[i][0] = 2 * boundary_values[i][0] - phi[i][1]; }
            if (boundary_type[i][0] == NEUMANN) { phi[i][0] = phi[i][1] - dy * boundary_values[i][0]; }
            // 上边界
            if (boundary_type[i][Ny + 1] == DIRICHLET) {
                phi[i][Ny + 1] = 2 * boundary_values[i][Ny + 1] - phi[i][Ny];
                // phi[i][Ny+1] = 2.0 - phi[i][Ny];
            }
            if (boundary_type[i][Ny + 1] == NEUMANN) { phi[i][Ny + 1] = phi[i][Ny] + dy * boundary_values[i][Ny + 1]; }
        }

        for (size_t j = 1; j < Ny + 1; j++) {
            // 左边界
            if (boundary_type[0][j] == DIRICHLET) { phi[0][j] = boundary_values[0][j]; }
            if (boundary_type[0][j] == NEUMANN) {
                double phi_ghost = phi[1][j] - 2 * dx * boundary_values[0][j];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi[1][j] + phi_ghost) / dx / dx + (phi[0][j + 1] + phi[0][j - 1]) / dy / dy;
                phi[0][j]        = (ub[0][j] + mu / rho * p2) / p1;
            }
            // 右边界
            if (boundary_type[Nx][j] == DIRICHLET) { phi[Nx][j] = boundary_values[Nx][j]; }
            if (boundary_type[Nx][j] == NEUMANN) {
                double phi_ghost = phi[Nx - 1][j] + 2 * dx * boundary_values[Nx][j];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi_ghost + phi[Nx - 1][j]) / dx / dx + (phi[Nx][j + 1] + phi[Nx][j - 1]) / dy / dy;
                phi[Nx][j]       = (ub[Nx][j] + mu / rho * p2) / p1;
            }
        }

        for (size_t i = 1; i < Nx; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double p1 = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2 = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                phi[i][j] = (ub[i][j] + mu / rho * p2) / p1;
            }
        }

        for (size_t i = 1; i < Nx; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double p1      = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2      = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                residual[i][j] = p1 * phi[i][j] - (ub[i][j] + mu / rho * p2);
            }
        }

        // 计算L2误差和L1误差
        double residual_norm_L2 = 0.0;
        double residual_norm_L1 = 0.0;

        for (size_t i = 0; i < Nx + 1; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                residual_norm_L1 = std::max(residual_norm_L1, std::abs(residual[i][j]));
                residual_norm_L2 += residual[i][j] * residual[i][j];
            }
        }
        residual_norm_L2 = std::sqrt(residual_norm_L2 * dx * dy);
        // if (iter % 1000 == 0)
        //     printf("iter: %08zu, Error L1 : %.16e, L2 : %.16e.\n", iter,
        //     residual_norm_L1, residual_norm_L2);

        // 跳出循环
        if (residual_norm_L2 < EPSILON) break;
    }
    return 0;
}

int tentitive_velocity_u(const std::vector<std::vector<double>>& f,  //(Nx+1)*(Ny+2)
                         const std::vector<std::vector<double>>& un, //(Nx+1)*(Ny+2)
                         std::vector<std::vector<double>>& phi, const std::vector<std::vector<int>>& boundary_type,
                         const std::vector<std::vector<double>>& boundary_values, double width, double height, double T,
                         double rho, double mu, size_t max_iters, size_t Nx, size_t Ny, size_t Nt) {
    // TODO: advection term hasn't been implemented yet.
    double dt = T / Nt;
    auto   ub = make_vector_2D<double>(Nx + 1, Ny + 2);
    for (size_t i = 0; i < Nx + 1; i++) {
        for (size_t j = 0; j < Ny + 2; j++) {
            ub[i][j] = un[i][j] / dt + f[i][j] / rho;
        }
    }
    tentitive_velocity_u(ub, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters, Nx, Ny, Nt);
    return 0;
}
#endif