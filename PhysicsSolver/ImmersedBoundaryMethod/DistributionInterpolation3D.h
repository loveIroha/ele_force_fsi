/// @date 2023-10-19
/// @file DistributionInterpolation3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <io.h>
#include <vector_functions.h>

#include <cmath>

#include "DistributionInterpolation3D_kokkos.h"

// TODO : This file should be reviewed and refactored carafully!
namespace interactor {
template <int N, typename T>
void outer_product(const T* v1, const T* v2, T* out) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            out[N * j + i] = v1[i] * v2[j];
        }
}

template <int N, typename T>
void outer_product(const T* v1, const T* v2, const T* v3, T* out) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                out[N * N * k + N * j + i] = v1[i] * v2[j] * v3[k];
            }
}

// Standard 4-point Kernel for IBM
double phi_ib4(double r) {
    r         = fabs(r);
    double r2 = r * r;
    double phi;
    if (r < 1) {
        phi = (3 - 2 * r + sqrt(1 + 4 * r - 4 * r2)) / 8.;
    } else if (r >= 1 && r < 2) {
        phi = (5 - 2 * r - sqrt(-7 + 12 * r - 4 * r2)) / 8.;
    } else {
        phi = 0.;
    }
    return phi;
}

template <int N, typename T>
void phi_function_weights(T* v, const T* r) {
    for (size_t i = 0; i < N; i++) {
        v[i] = phi_ib4(r[i]);
    }
}

template <int N, typename T>
void delta_function_weights(T* out, const T& r1, const T& r2, const T& r3) {
    const double r1_v[N] = {-1 - r1, -r1, 1 - r1, 2 - r1};
    const double r2_v[N] = {-1 - r2, -r2, 1 - r2, 2 - r2};
    const double r3_v[N] = {-1 - r3, -r3, 1 - r3, 2 - r3};

    double v1[N] = {};
    double v2[N] = {};
    double v3[N] = {};

    phi_function_weights<N>(v1, r1_v);
    phi_function_weights<N>(v2, r2_v);
    phi_function_weights<N>(v3, r3_v);

    outer_product<N>(v1, v2, v3, out);
}

/// @brief
///
/// @param lagrange_u   Lagrange 速度分量 u
/// @param eulerian_u   Eulerian 速度分量 u
/// @param num          Lagrange 点的数量
/// @param dim          Eulerian 网格大小
/// @param h            Eulerian 网格密度
/// @param quadrature_rules
///
void interpolate_u(double* lagrange_u, const double* eulerian_u, int num, int Nx, int Ny, int Nz, double hx, double hy,
                   double hz, const double4* quadrature_rules) {
    ScopeProfiler _{__func__};

    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];

        // i,j 为 u 在数组中的索引，而不是所在单元格的索引
        // dim 为 u 的数量，而不是单元格的数量
        // cell[i,j] u[i,j+1], [i*hx,j*hy+1/2]
        double x  = (pos.x + hx) / hx;
        double y  = (pos.y + 0.5 * hy) / hy;
        double z  = (pos.z + 0.5 * hz) / hz;
        int    i  = floor(x);
        int    j  = floor(y);
        int    k  = floor(z);
        double r1 = x - i;
        double r2 = y - j;
        double r3 = z - k;

        double w[64] = {};
        double u[64] = {};
        delta_function_weights<4>(w, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx                                   = (ii + i) + Nx * (jj + j) + Ny * Nx * (kk + k);
                    u[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_u[ridx];
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += u[i] * w[i];

        lagrange_u[gidx] = sum;
    }
}

void interpolate_v(double* lagrange_v, const double* eulerian_v, int num, int Nx, int Ny, int Nz, double hx, double hy,
                   double hz, const double4* quadrature_rules) {
    ScopeProfiler _{__func__};
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = (pos.x + 0.5 * hx) / hx;
        double  y   = (pos.y + hy) / hy;
        double  z   = (pos.z + 0.5 * hz) / hz;
        int     i   = floor(x);
        int     j   = floor(y);
        int     k   = floor(z);
        double  r1  = x - i;
        double  r2  = y - j;
        double  r3  = z - k;

        double w[64] = {};
        double v[64] = {};
        delta_function_weights<4>(w, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx                                   = (ii + i) + Nx * (jj + j) + Ny * Nx * (kk + k);
                    v[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_v[ridx];
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += v[i] * w[i];

        lagrange_v[gidx] = sum;
    }
}

void interpolate_w(double* lagrange_w, const double* eulerian_w, int num, int Nx, int Ny, int Nz, double hx, double hy,
                   double hz, const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);

    ScopeProfiler _{__func__};
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = (pos.x + 0.5 * hx) / hx;
        double  y   = (pos.y + 0.5 * hy) / hy;
        double  z   = (pos.z + hz) / hz;
        int     i   = floor(x);
        int     j   = floor(y);
        int     k   = floor(z);
        double  r1  = x - i;
        double  r2  = y - j;
        double  r3  = z - k;

        double delta_weight[64] = {};
        double w[64]            = {};
        delta_function_weights<4>(delta_weight, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx                                   = (ii + i) + Nx * (jj + j) + Nx * Ny * (kk + k);
                    w[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_w[ridx];
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += w[i] * delta_weight[i];

        lagrange_w[gidx] = sum;
    }
}

void distribute_u(const double* lagrange_u, double* eulerian_u, int num, int Nx, int Ny, int Nz, double hx, double hy,
                  double hz, const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);
    ScopeProfiler _{__func__};
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = (pos.x + hx) / hx;
        double  y   = (pos.y + 0.5 * hy) / hy;
        double  z   = (pos.z + 0.5 * hz) / hz;
        int     i   = floor(x);
        int     j   = floor(y);
        int     k   = floor(z);
        double  r1  = x - i;
        double  r2  = y - j;
        double  r3  = z - k;

        double delta_weight[64] = {};
        double f[64]            = {};
        delta_function_weights<4>(delta_weight, r1, r2, r3);

        double F      = lagrange_u[gidx];
        double inv_h3 = 1.0 / hx / hy / hz;

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * delta_weight[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx = (ii + i) + (jj + j) * Nx + (kk + k) * Nx * Ny;
                    eulerian_u[ridx] += f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)];
                    // LOG_F(INFO, "ridx: %d, f: %f", ridx, f[16 * (kk + 1) + 4
                    // * (jj + 1) + (ii + 1)]);
                }
            }
        }
    }
}

void distribute_v(const double* lagrange_v, double* eulerian_v, int num, int Nx, int Ny, int Nz, double hx, double hy,
                  double hz, const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);
    ScopeProfiler _{__func__};
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = (pos.x + 0.5 * hx) / hx;
        double  y   = (pos.y + hy) / hy;
        double  z   = (pos.z + 0.5 * hz) / hz;
        int     i   = floor(x);
        int     j   = floor(y);
        int     k   = floor(z);
        double  r1  = x - i;
        double  r2  = y - j;
        double  r3  = z - k;

        double delta_weight[64] = {};
        double f[64]            = {};
        delta_function_weights<4>(delta_weight, r1, r2, r3);

        double F      = lagrange_v[gidx];
        double inv_h3 = 1.0 / hx / hy / hz;

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * delta_weight[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx = (ii + i) + (jj + j) * Nx + (kk + k) * Nx * Ny;
                    eulerian_v[ridx] += f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)];
                }
            }
        }
    }
}

void distribute_w(const double* lagrange_w, double* eulerian_w, int num, int Nx, int Ny, int Nz, double hx, double hy,
                  double hz, const double4* quadrature_rules) {
    ScopeProfiler _{__func__};
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = (pos.x + 0.5 * hx) / hx;
        double  y   = (pos.y + 0.5 * hy) / hy;
        double  z   = (pos.z + hz) / hz;
        int     i   = floor(x);
        int     j   = floor(y);
        int     k   = floor(z);
        double  r1  = x - i;
        double  r2  = y - j;
        double  r3  = z - k;

        double delta_weight[64] = {};
        double f[64]            = {};
        delta_function_weights<4>(delta_weight, r1, r2, r3);

        double F      = lagrange_w[gidx];
        double inv_h3 = 1.0 / hx / hy / hz;

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * delta_weight[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (jj + j) < Ny && (ii + i) >= 0 && (ii + i) < Nx && (kk + k) >= 0
                          && (kk + k) < Nz)) {
                        continue;
                    }
                    int ridx = (ii + i) + (jj + j) * Nx + (kk + k) * Nx * Ny;
                    eulerian_w[ridx] += f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)];
                }
            }
        }
    }
}

} // namespace interactor