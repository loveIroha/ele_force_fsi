/// @date 2023-12-09
/// @file DistributionInterpolation3D_kokkos.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief kokkos version
///
///

#pragma once

#include "DistributionInterpolation2D_kokkos.h"

namespace mykokkos {

template <typename T>
KOKKOS_INLINE_FUNCTION void outer_product(const T* v1, const T* v2, const T* v3, T* out, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                out[N * N * k + N * j + i] = v1[i] * v2[j] * v3[k];
            }
}

template <typename T, typename PHI = PhiIB4>
KOKKOS_INLINE_FUNCTION void delta_function_weights(T* out, const T& r1, const T& r2, const T& r3) {
    PHI    phi_ib4;
    double v1[PHI::N]   = {};
    double v2[PHI::N]   = {};
    double v3[PHI::N]   = {};
    double r1_v[PHI::N] = {};
    double r2_v[PHI::N] = {};
    double r3_v[PHI::N] = {};

    local_distance<PHI::N>(r1_v, r1);
    local_distance<PHI::N>(r2_v, r2);
    local_distance<PHI::N>(r3_v, r3);

    phi_function_weights(v1, r1_v, phi_ib4);
    phi_function_weights(v2, r2_v, phi_ib4);
    phi_function_weights(v3, r3_v, phi_ib4);

    outer_product(v1, v2, v3, out, PHI::N);
}

struct LocalInterpolate_u_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_u;
    ViewMatrixType  eulerian_u;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalInterpolate_u_3D(const double* eulerian_u_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                          int num)
        : lagrange_u("lagrange_u", num), eulerian_u("eulerian_u", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        LOG_SCOPE_FUNCTION(INFO);
        ViewMatrixType::HostMirror  eulerian_u_h       = Kokkos::create_mirror_view(eulerian_u);
        ViewDouble4Type::HostMirror quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < dim.x; i++) {
            for (size_t j = 0; j < dim.y; j++) {
                for (size_t k = 0; k < dim.z; k++) {
                    eulerian_u_h(i, j, k) = eulerian_u_raw[i + j * dim.x + k * dim.x * dim.y];
                }
            }
        }

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
        }

        Kokkos::deep_copy(eulerian_u, eulerian_u_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        LOG_SCOPE_FUNCTION(INFO);
        ViewVectorType::HostMirror lagrange_u_h = Kokkos::create_mirror_view(lagrange_u);
        Kokkos::deep_copy(lagrange_u_h, lagrange_u);

        for (size_t i = 0; i < num; i++) {
            result[i] = lagrange_u_h(i);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos = quadrature_rules(gidx);

        double x     = (pos.x + h.x) / h.x;
        double y     = (pos.y + 0.5 * h.y) / h.y;
        double z     = (pos.z + 0.5 * h.z) / h.z;
        int    i     = floor(x);
        int    j     = floor(y);
        int    k     = floor(z);
        double r1    = x - i;
        double r2    = y - j;
        double r3    = z - k;
        double w[64] = {};
        double u[64] = {};

        delta_function_weights(w, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    u[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_u(ii + i, jj + j, kk + k);
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += u[i] * w[i];

        lagrange_u[gidx] = sum;
    }
};

struct LocalInterpolate_v_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_v;
    ViewMatrixType  eulerian_v;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalInterpolate_v_3D(const double* eulerian_v_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                          int num)
        : lagrange_v("lagrange_u", num), eulerian_v("eulerian_u", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        ViewMatrixType::HostMirror  eulerian_v_h       = Kokkos::create_mirror_view(eulerian_v);
        ViewDouble4Type::HostMirror quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < dim.x; i++) {
            for (size_t j = 0; j < dim.y; j++) {
                for (size_t k = 0; k < dim.z; k++) {
                    eulerian_v_h(i, j, k) = eulerian_v_raw[i + j * dim.x + k * dim.x * dim.y];
                }
            }
        }

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
        }

        Kokkos::deep_copy(eulerian_v, eulerian_v_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        ViewVectorType::HostMirror lagrange_v_h = Kokkos::create_mirror_view(lagrange_v);
        Kokkos::deep_copy(lagrange_v_h, lagrange_v);

        for (size_t i = 0; i < num; i++) {
            result[i] = lagrange_v_h(i);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos = quadrature_rules(gidx);

        double x     = (pos.x + 0.5 * h.x) / h.x;
        double y     = (pos.y + h.y) / h.y;
        double z     = (pos.z + 0.5 * h.z) / h.z;
        int    i     = floor(x);
        int    j     = floor(y);
        int    k     = floor(z);
        double r1    = x - i;
        double r2    = y - j;
        double r3    = z - k;
        double w[64] = {};
        double v[64] = {};

        delta_function_weights(w, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    v[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_v(ii + i, jj + j, kk + k);
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += v[i] * w[i];

        lagrange_v[gidx] = sum;
    }
};

struct LocalInterpolate_w_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_w;
    ViewMatrixType  eulerian_w;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalInterpolate_w_3D(const double* eulerian_w_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                          int num)
        : lagrange_w("lagrange_u", num), eulerian_w("eulerian_u", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        ViewMatrixType::HostMirror  eulerian_w_h       = Kokkos::create_mirror_view(eulerian_w);
        ViewDouble4Type::HostMirror quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < dim.x; i++) {
            for (size_t j = 0; j < dim.y; j++) {
                for (size_t k = 0; k < dim.z; k++) {
                    eulerian_w_h(i, j, k) = eulerian_w_raw[i + j * dim.x + k * dim.x * dim.y];
                }
            }
        }

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
        }

        Kokkos::deep_copy(eulerian_w, eulerian_w_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        ViewVectorType::HostMirror lagrange_w_h = Kokkos::create_mirror_view(lagrange_w);
        Kokkos::deep_copy(lagrange_w_h, lagrange_w);

        for (size_t i = 0; i < num; i++) {
            result[i] = lagrange_w_h(i);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos = quadrature_rules(gidx);

        double x      = (pos.x + 0.5 * h.x) / h.x;
        double y      = (pos.y + 0.5 * h.y) / h.y;
        double z      = (pos.z + h.z) / h.z;
        int    i      = floor(x);
        int    j      = floor(y);
        int    k      = floor(z);
        double r1     = x - i;
        double r2     = y - j;
        double r3     = z - k;
        double w[64]  = {};
        double ww[64] = {};

        delta_function_weights(w, r1, r2, r3);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    ww[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)] = eulerian_w(ii + i, jj + j, kk + k);
                }
            }
        }

        double sum{};
        for (size_t i = 0; i < 64; i++)
            sum += ww[i] * w[i];

        lagrange_w[gidx] = sum;
    }
};

struct LocalDistribute_u_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_u;
    ViewMatrixType  eulerian_u;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalDistribute_u_3D(const double* lagrange_u_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                         int num)
        : lagrange_u("lagrange_u", num), eulerian_u("eulerian_u", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        auto lagrange_u_h       = Kokkos::create_mirror_view(lagrange_u);
        auto quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
            lagrange_u_h(i)       = lagrange_u_raw[i];
        }

        Kokkos::deep_copy(lagrange_u, lagrange_u_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        auto eulerian_u_h = Kokkos::create_mirror_view(eulerian_u);
        Kokkos::deep_copy(eulerian_u_h, eulerian_u);

        for (size_t i = 0; i < dim.x; i++)
            for (size_t j = 0; j < dim.y; j++)
                for (size_t k = 0; k < dim.z; k++) {
                    result[i + j * dim.x + k * dim.x * dim.y] = eulerian_u_h(i, j, k);
                }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos    = quadrature_rules(gidx);
        double  x      = (pos.x + h.x) / h.x;
        double  y      = (pos.y + 0.5 * h.y) / h.y;
        double  z      = (pos.z + 0.5 * h.z) / h.z;
        int     i      = floor(x);
        int     j      = floor(y);
        int     k      = floor(z);
        double  r1     = x - i;
        double  r2     = y - j;
        double  r3     = z - k;
        double  F      = lagrange_u[gidx];
        double  inv_h3 = 1.0 / h.x / h.y / h.z;
        double  w[64]  = {};
        double  f[64]  = {};

        delta_function_weights(w, r1, r2, r3);

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    Kokkos::atomic_add(&eulerian_u(ii + i, jj + j, kk + k), f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)]);
                }
            }
        }
    }
};

struct LocalDistribute_v_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_v;
    ViewMatrixType  eulerian_v;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalDistribute_v_3D(const double* lagrange_v_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                         int num)
        : lagrange_v("lagrange_v", num), eulerian_v("eulerian_v", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        auto lagrange_v_h       = Kokkos::create_mirror_view(lagrange_v);
        auto quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
            lagrange_v_h(i)       = lagrange_v_raw[i];
        }

        Kokkos::deep_copy(lagrange_v, lagrange_v_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        auto eulerian_v_h = Kokkos::create_mirror_view(eulerian_v);
        Kokkos::deep_copy(eulerian_v_h, eulerian_v);

        for (size_t i = 0; i < dim.x; i++)
            for (size_t j = 0; j < dim.y; j++)
                for (size_t k = 0; k < dim.z; k++) {
                    result[i + j * dim.x + k * dim.x * dim.y] = eulerian_v_h(i, j, k);
                }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos    = quadrature_rules(gidx);
        double  x      = (pos.x + 0.5 * h.x) / h.x;
        double  y      = (pos.y + h.y) / h.y;
        double  z      = (pos.z + 0.5 * h.z) / h.z;
        int     i      = floor(x);
        int     j      = floor(y);
        int     k      = floor(z);
        double  r1     = x - i;
        double  r2     = y - j;
        double  r3     = z - k;
        double  F      = lagrange_v[gidx];
        double  inv_h3 = 1.0 / h.x / h.y / h.z;
        double  w[64]  = {};
        double  f[64]  = {};

        delta_function_weights(w, r1, r2, r3);

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    Kokkos::atomic_add(&eulerian_v(ii + i, jj + j, kk + k), f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)]);
                }
            }
        }
    }
};
struct LocalDistribute_w_3D {
    typedef Kokkos::View<double*>   ViewVectorType;
    typedef Kokkos::View<double***> ViewMatrixType;
    typedef Kokkos::View<double4*>  ViewDouble4Type;

    ViewVectorType  lagrange_w;
    ViewMatrixType  eulerian_w;
    ViewDouble4Type quadrature_rules;

    // These variables should be moved to the GPU?
    double3 h;
    int3    dim;
    int     num;

    LocalDistribute_w_3D(const double* lagrange_w_raw, const double4* quadrature_rules_raw, double3 h, int3 dim,
                         int num)
        : lagrange_w("lagrange_u", num), eulerian_w("eulerian_u", dim.x, dim.y, dim.z),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        LOG_SCOPE_FUNCTION(INFO);

        auto lagrange_w_h       = Kokkos::create_mirror_view(lagrange_w);
        auto quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
            lagrange_w_h(i)       = lagrange_w_raw[i];
        }

        Kokkos::deep_copy(lagrange_w, lagrange_w_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        LOG_SCOPE_FUNCTION(INFO);
        auto eulerian_w_h = Kokkos::create_mirror_view(eulerian_w);
        Kokkos::deep_copy(eulerian_w_h, eulerian_w);

        for (size_t i = 0; i < dim.x; i++)
            for (size_t j = 0; j < dim.y; j++)
                for (size_t k = 0; k < dim.z; k++) {
                    result[i + j * dim.x + k * dim.x * dim.y] = eulerian_w_h(i, j, k);
                }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos    = quadrature_rules(gidx);
        double  x      = (pos.x + 0.5 * h.x) / h.x;
        double  y      = (pos.y + 0.5 * h.y) / h.y;
        double  z      = (pos.z + h.z) / h.z;
        int     i      = floor(x);
        int     j      = floor(y);
        int     k      = floor(z);
        double  r1     = x - i;
        double  r2     = y - j;
        double  r3     = z - k;
        double  F      = lagrange_w[gidx];
        double  inv_h3 = 1.0 / h.x / h.y / h.z;
        double  w[64]  = {};
        double  f[64]  = {};

        delta_function_weights(w, r1, r2, r3);

        for (size_t i = 0; i < 64; i++) {
            f[i] = inv_h3 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                for (int kk = -1; kk <= 2; kk++) {
                    if (!((jj + j) >= 0 && (kk + k) >= 0 && (ii + i) >= 0 && (jj + j) < dim.y && (ii + i) < dim.x
                          && (kk + k) < dim.z)) {
                        continue;
                    }
                    Kokkos::atomic_add(&eulerian_w(ii + i, jj + j, kk + k), f[16 * (kk + 1) + 4 * (jj + 1) + (ii + 1)]);
                }
            }
        }
    }
};

} // namespace mykokkos

void interpolate_u(double* lagrange_u, const double* eulerian_u, const double4* quadrature_rules, double3 h, int3 dim,
                   int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalInterpolate_u_3D local_interpolate(eulerian_u, quadrature_rules, h, dim, num);

    // 并行计算
    Kokkos::parallel_for(local_interpolate.num, local_interpolate);
    local_interpolate.extract_result(lagrange_u);
}

void interpolate_v(double* lagrange_v, const double* eulerian_v, const double4* quadrature_rules, double3 h, int3 dim,
                   int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalInterpolate_v_3D local_interpolate(eulerian_v, quadrature_rules, h, dim, num);
    // 并行计算
    Kokkos::parallel_for(local_interpolate.num, local_interpolate);
    local_interpolate.extract_result(lagrange_v);
}
void interpolate_w(double* lagrange_w, const double* eulerian_w, const double4* quadrature_rules, double3 h, int3 dim,
                   int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalInterpolate_w_3D local_interpolate(eulerian_w, quadrature_rules, h, dim, num);
    // 并行计算
    Kokkos::parallel_for(local_interpolate.num, local_interpolate);
    local_interpolate.extract_result(lagrange_w);
}

void distribute_u(double* eulerian_u, const double* lagrange_u, const double4* quadrature_rules, double3 h, int3 dim,
                  int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalDistribute_u_3D local_distribute(lagrange_u, quadrature_rules, h, dim, num);
    Kokkos::parallel_for(local_distribute.num, local_distribute);
    local_distribute.extract_result(eulerian_u);
}
void distribute_v(double* eulerian_v, const double* lagrange_v, const double4* quadrature_rules, double3 h, int3 dim,
                  int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalDistribute_v_3D local_distribute(lagrange_v, quadrature_rules, h, dim, num);
    Kokkos::parallel_for(local_distribute.num, local_distribute);
    local_distribute.extract_result(eulerian_v);
}

void distribute_w(double* eulerian_w, const double* lagrange_w, const double4* quadrature_rules, double3 h, int3 dim,
                  int num) {
    LOG_SCOPE_FUNCTION(INFO);
    mykokkos::LocalDistribute_w_3D local_distribute(lagrange_w, quadrature_rules, h, dim, num);
    Kokkos::parallel_for(local_distribute.num, local_distribute);
    local_distribute.extract_result(eulerian_w);
}
