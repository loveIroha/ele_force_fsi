/// @date 2023-10-04
/// @file Projection3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __PROJECTION_3D_H__
#define __PROJECTION_3D_H__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

namespace projection_3D {
const int DIM = 3;

using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

double calculate_velocity_divergence(const MultiArrayDouble& u, // (Nx+1)*(Ny+2)*(Nz+2)
                                     const MultiArrayDouble& v, // (Nx+2)*(Ny+1)*(Nz+2)
                                     const MultiArrayDouble& w, // (Nx+2)*(Ny+2)*(Nz+1)
                                     int Nx, int Ny, int Nz, double width, double height, double depth) {
    double div = 0.0;

    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                div += ((u[i + 2][j + 1][k + 1] - u[i + 1][j + 1][k + 1]) / dx
                        + (v[i + 1][j + 2][k + 1] - v[i + 1][j + 1][k + 1]) / dy
                        + (w[i + 1][j + 1][k + 2] - w[i + 1][j + 1][k + 1]) / dz);
            }
        }
    }

    return div * dx * dy * dz;
}
// [[deprecated("This function is deprecated. Use new_function() instead.")]]
int make_pb(MultiArrayDouble&       pb, // (Nx+2)*(Ny+2)*(Nz+2)
            const MultiArrayDouble& u,  // (Nx+1)*(Ny+2)*(Nz+2)
            const MultiArrayDouble& v,  // (Nx+2)*(Ny+1)*(Nz+2)
            const MultiArrayDouble& w,  // (Nx+2)*(Ny+2)*(Nz+1)
            int Nx, int Ny, int Nz, double dt, double width, double height, double depth, double rho,
            double mu) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                pb[i + 1][j + 1][k + 1] = rho / dt
                                          * ((u[i + 2][j + 1][k + 1] - u[i + 1][j + 1][k + 1]) / dx
                                             + (v[i + 1][j + 2][k + 1] - v[i + 1][j + 1][k + 1]) / dy
                                             + (w[i + 1][j + 1][k + 2] - w[i + 1][j + 1][k + 1]) / dz);
            }
        }
    }

    return 0;
}

int make_pb(MultiArrayDouble&       pb, // (Nx+2)*(Ny+2)*(Nz+2)
            const MultiArrayDouble& s,  // (Nx+2)*(Ny+2)*(Nz+2)
            const MultiArrayDouble& u,  // (Nx+1)*(Ny+2)*(Nz+2)
            const MultiArrayDouble& v,  // (Nx+2)*(Ny+1)*(Nz+2)
            const MultiArrayDouble& w,  // (Nx+2)*(Ny+2)*(Nz+1)
            double width, double height, double depth, double dt, int Nx, int Ny, int Nz, double rho,
            double mu) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                pb[i + 1][j + 1][k + 1]
                    = rho / dt
                      * ((u[i + 1][j + 1][k + 1] - u[i][j + 1][k + 1]) / dx
                         + (v[i + 1][j + 1][k + 1] - v[i + 1][j][k + 1]) / dy
                         + (w[i + 1][j + 1][k + 1] - w[i + 1][j + 1][k]) / dz - s[i + 1][j + 1][k + 1]);
            }
        }
    }

    return 0;
}

void correct_velocity_u(MultiArrayDouble& u, const MultiArrayDouble& u_, const MultiArrayDouble& p, double width,
                        double height, double depth, double dt, int Nx, int Ny, int Nz, double rho) {
    [[maybe_unused]] double dx = width / Nx;
    [[maybe_unused]] double dy = height / Ny;
    [[maybe_unused]] double dz = depth / Nz;

    for (int i = 0; i < Nx + 1; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                u[i + 1][j + 1][k + 1]
                    = u_[i + 1][j + 1][k + 1] - dt / rho / dx * (p[i + 1][j + 1][k + 1] - p[i][j + 1][k + 1]);
            }
            // 边界外虚拟点的值
            // u[i][0] = u_[i][0];
            // u[i][_Ny + 1] = u_[i][_Ny + 1];
        }
    }
}

void correct_velocity_v(MultiArrayDouble& v, const MultiArrayDouble& v_, const MultiArrayDouble& p, double width,
                        double height, double depth, double dt, int Nx, int Ny, int Nz, double rho) {
    [[maybe_unused]] double dx = width / Nx;
    [[maybe_unused]] double dy = height / Ny;
    [[maybe_unused]] double dz = depth / Nz;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny + 1; j++) {
            for (int k = 0; k < Nz; k++) {
                v[i + 1][j + 1][k + 1]
                    = v_[i + 1][j + 1][k + 1] - dt / rho / dy * (p[i + 1][j + 1][k + 1] - p[i + 1][j][k + 1]);
            }
            // 边界外虚拟点的值
            // u[i][0] = u_[i][0];
            // u[i][_Ny + 1] = u_[i][_Ny + 1];
        }
    }
}

void correct_velocity_w(MultiArrayDouble& w, const MultiArrayDouble& w_, const MultiArrayDouble& p, double width,
                        double height, double depth, double dt, int Nx, int Ny, int Nz, double rho) {
    [[maybe_unused]] double dx = width / Nx;
    [[maybe_unused]] double dy = height / Ny;
    [[maybe_unused]] double dz = depth / Nz;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz + 1; k++) {
                w[i + 1][j + 1][k + 1]
                    = w_[i + 1][j + 1][k + 1] - dt / rho / dz * (p[i + 1][j + 1][k + 1] - p[i + 1][j + 1][k]);
            }
            // 边界外虚拟点的值
            // u[i][0] = u_[i][0];
            // u[i][_Ny + 1] = u_[i][_Ny + 1];
        }
    }
}
void correct_velocity(MultiArrayDouble& u, MultiArrayDouble& v, MultiArrayDouble& w, const MultiArrayDouble& u_,
                      const MultiArrayDouble& v_, const MultiArrayDouble& w_, const MultiArrayDouble& p, int Nx, int Ny,
                      int Nz, double dt, double width, double height, double depth, double rho) {
    ScopeProfiler _{__func__};
    correct_velocity_u(u, u_, p, width, height, depth, dt, Nx, Ny, Nz, rho);
    correct_velocity_v(v, v_, p, width, height, depth, dt, Nx, Ny, Nz, rho);
    correct_velocity_w(w, w_, p, width, height, depth, dt, Nx, Ny, Nz, rho);
}
} // namespace projection_3D

#endif