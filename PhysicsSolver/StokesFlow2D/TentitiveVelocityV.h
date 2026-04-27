#include "header.h"
#ifndef __TENTITIVE_VELOCITY_V_H__
#define __TENTITIVE_VELOCITY_V_H__

int tentitive_velocity_v(const std::vector<std::vector<double>>& vb, //(Nx+1)*(Ny+2)
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

    auto residual = make_vector_2D<double>(Nx + 2, Ny + 1);

    for (size_t iter = 0; iter < max_iters; iter++) {
        for (size_t i = 1; i < Nx + 1; i++) {
            // 下边界
            if (boundary_type[i][0] == DIRICHLET) { phi[i][0] = boundary_values[i][0]; }
            if (boundary_type[i][0] == NEUMANN) {
                double ghost_phi = phi[i][1] - 2.0 * dy * boundary_values[i][0];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi[i + 1][0] + phi[i - 1][0]) / dx / dx + (phi[i][1] + ghost_phi) / dy / dy;
                phi[i][0]        = (vb[i][0] + mu / rho * p2) / p1;
            }
            // 上边界
            if (boundary_type[i][Ny] == DIRICHLET) { phi[i][Ny] = boundary_values[i][Ny]; }
            if (boundary_type[i][Ny] == NEUMANN) {
                double ghost_phi = phi[i][Ny - 1] + 2.0 * dy * boundary_values[i][Ny];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi[i + 1][Ny] + phi[i - 1][Ny]) / dx / dx + (ghost_phi + phi[i][Ny - 1]) / dy / dy;
                phi[i][Ny]       = (vb[i][Ny] + mu / rho * p2) / p1;
            }
        }

        for (size_t j = 0; j < Ny + 1; j++) {
            // 左边界
            if (boundary_type[0][j] == DIRICHLET) { phi[0][j] = 2 * boundary_values[0][j] - phi[1][j]; }
            if (boundary_type[0][j] == NEUMANN) { phi[0][j] = phi[1][j] - dy * boundary_values[0][j]; }
            // 右边界
            if (boundary_type[Nx + 1][j] == DIRICHLET) { phi[Nx + 1][j] = 2 * boundary_values[Nx + 1][j] - phi[Nx][j]; }
            if (boundary_type[Nx + 1][j] == NEUMANN) { phi[Nx + 1][j] = phi[Nx][j] + dy * boundary_values[Nx + 1][j]; }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny; j++) {
                double p1 = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2 = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                phi[i][j] = (vb[i][j] + mu / rho * p2) / p1;
            }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny; j++) {
                // double p1 = (phi[i - 1][j] - 2 * phi[i][j] + phi[i + 1][j]) /
                // dx / dx; double p2 = (phi[i][j - 1] - 2 * phi[i][j] + phi[i][j
                // + 1]) / dy / dy; residual[i][j] = vb[i][j] - phi[i][j] / dt +
                // mu / rho * (p1 + p2);
                double p1      = +2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2      = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                residual[i][j] = -phi[i][j] * (p1 + 1.0 / dt) + vb[i][j] + (mu / rho * p2);
            }
        }

        // 计算L2误差和L1误差
        double residual_norm_L2 = 0.0;
        double residual_norm_L1 = 0.0;

        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 1; j++) {
                residual_norm_L1 = std::max(residual_norm_L1, std::abs(residual[i][j]));
                residual_norm_L2 += residual[i][j] * residual[i][j];
            }
        }
        residual_norm_L2 = std::sqrt(residual_norm_L2 * dx * dy);
        // if (iter%1000==0)
        // printf("iter: %08zu, Error L1 : %.16e, L2 : %.16e.\n", iter,
        // residual_norm_L1, residual_norm_L2);

        // 跳出循环
        if (residual_norm_L2 < EPSILON) break;
    }
    return 0;
}

int tentitive_velocity_v(const std::vector<std::vector<double>>& f,  //(Nx+2)*(Ny+1)
                         const std::vector<std::vector<double>>& vn, //(Nx+2)*(Ny+1)
                         std::vector<std::vector<double>>& phi, const std::vector<std::vector<int>>& boundary_type,
                         const std::vector<std::vector<double>>& boundary_values, double width, double height, double T,
                         double rho, double mu, size_t max_iters, size_t Nx, size_t Ny, size_t Nt) {
    // TODO: advection term hasn't been implemented yet.
    double dt = T / Nt;
    auto   vb = make_vector_2D<double>(Nx + 2, Ny + 1);

    LOG_F(INFO, "Computing tentitive velocity v.");
    LOG_F(INFO, "Nx    : %05zu     , Ny     : %05zu     , Nt : %05zu     , max_iters : %05zu.", Nx, Ny, Nt, max_iters);
    for (size_t i = 0; i < Nx + 2; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            vb[i][j] = vn[i][j] / dt + f[i][j] / rho;
        }
    }
    tentitive_velocity_v(vb, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters, Nx, Ny, Nt);
    return 0;
}
#endif