/// @date 2023-12-01
/// @file DistributionInterpolation2D_kokkos.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <boost/multi_array.hpp>

#include <Kokkos_Core.hpp>
#include <io/loguru.hpp>
#include <io/writeVTK.h>
#include <vector_functions.h>

#include <cmath>

// TODO : This file should be reviewed and refactored carafully!
namespace mykokkos {

template <typename T>
KOKKOS_INLINE_FUNCTION void outer_product(const T* v1, const T* v2, T* out, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            out[N * j + i] = v1[i] * v2[j];
        }
}

// Four points Kernel for IBM
struct PhiIB4 {
    static constexpr int N = 4;

    KOKKOS_INLINE_FUNCTION double operator()(double r) {
        r = std::abs(r);
        double phi;
        double r2 = r * r;
        if (r < 1) {
            phi = (3 - 2 * r + sqrt(1 + 4 * r - 4 * r2)) / 8.;
        } else if (r >= 1 && r < 2) {
            phi = (5 - 2 * r - sqrt(-7 + 12 * r - 4 * r2)) / 8.;
        } else {
            phi = 0.;
        }
        return phi;
    }
};

template <typename T, typename Func>
KOKKOS_INLINE_FUNCTION void phi_function_weights(T* v, const T* r, Func phi) {
    for (size_t i = 0; i < phi.N; i++) {
        v[i] = phi(r[i]);
    }
}

template <int N>
KOKKOS_INLINE_FUNCTION void local_distance(double* rs, double r) {
    if constexpr (N == 4) {
        rs[0] = -1 - r;
        rs[1] = -r;
        rs[2] = 1 - r;
        rs[3] = 2 - r;
    } else {
        assert(false);
    }
}

template <typename T, typename PHI = PhiIB4>
KOKKOS_INLINE_FUNCTION void delta_function_weights(T* out, const T& r1, const T& r2) {
    PHI    phi_ib4;
    double v1[PHI::N]   = {};
    double v2[PHI::N]   = {};
    double r1_v[PHI::N] = {};
    double r2_v[PHI::N] = {};

    local_distance<PHI::N>(r1_v, r1);
    local_distance<PHI::N>(r2_v, r2);

    phi_function_weights(v1, r1_v, phi_ib4);
    phi_function_weights(v2, r2_v, phi_ib4);

    outer_product(v1, v2, out, phi_ib4.N);
}

struct LocalInterpolate_u {
    typedef Kokkos::View<double*>  ViewVectorType;
    typedef Kokkos::View<double**> ViewMatrixType;
    typedef Kokkos::View<double4*> ViewDouble4Type;

    Kokkos::View<double*>  lagrange_u;
    Kokkos::View<double**> eulerian_u;
    Kokkos::View<double4*> quadrature_rules;

    // These variables should be moved to the GPU?
    double2 h;
    int2    dim;
    int     num;

    LocalInterpolate_u(const double* eulerian_u_raw, const double4* quadrature_rules_raw, double2 h, int2 dim, int num)
        : lagrange_u("lagrange_u", num), eulerian_u("eulerian_u", dim.x, dim.y),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        ViewMatrixType::HostMirror  eulerian_u_h       = Kokkos::create_mirror_view(eulerian_u);
        ViewDouble4Type::HostMirror quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < dim.x; i++) {
            for (size_t j = 0; j < dim.y; j++) {
                eulerian_u_h(i, j) = eulerian_u_raw[i + j * dim.x];
            }
        }

        for (size_t i = 0; i < num; i++) {
            quadrature_rules_h(i) = quadrature_rules_raw[i];
        }

        Kokkos::deep_copy(eulerian_u, eulerian_u_h);
        Kokkos::deep_copy(quadrature_rules, quadrature_rules_h);
    }

    void extract_result(double* result) {
        ViewVectorType::HostMirror lagrange_u_h = Kokkos::create_mirror_view(lagrange_u);
        Kokkos::deep_copy(lagrange_u_h, lagrange_u);

        for (size_t i = 0; i < num; i++) {
            result[i] = lagrange_u_h(i);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos   = quadrature_rules(gidx);
        double  x     = pos.x / h.x;
        double  y     = (pos.y + 0.5 * h.y) / h.y;
        int     i     = floor(x);
        int     j     = floor(y);
        double  r1    = x - i;
        double  r2    = y - j;
        double  w[16] = {};
        double  u[16] = {};

        delta_function_weights(w, r1, r2);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) { continue; }
                u[4 * (jj + 1) + (ii + 1)] = eulerian_u(ii + i, jj + j);
            }
        }

        double sum{};
        for (size_t i = 0; i < 16; i++)
            sum += u[i] * w[i];

        lagrange_u(gidx) = sum;
    }
};

struct LocalInterpolate_v {
    typedef Kokkos::View<double*>  ViewVectorType;
    typedef Kokkos::View<double**> ViewMatrixType;
    typedef Kokkos::View<double4*> ViewDouble4Type;

    Kokkos::View<double*>  lagrange_v;
    Kokkos::View<double**> eulerian_v;
    Kokkos::View<double4*> quadrature_rules;

    // These variables should be moved to the GPU?
    double2 h;
    int2    dim;
    int     num;

    LocalInterpolate_v(const double* eulerian_v_raw, const double4* quadrature_rules_raw, double2 h, int2 dim, int num)
        : lagrange_v("lagrange_u", num), eulerian_v("eulerian_u", dim.x, dim.y),
          quadrature_rules("quadrature_rules", num), h(h), dim(dim), num(num) {
        ViewMatrixType::HostMirror  eulerian_v_h       = Kokkos::create_mirror_view(eulerian_v);
        ViewDouble4Type::HostMirror quadrature_rules_h = Kokkos::create_mirror_view(quadrature_rules);

        for (size_t i = 0; i < dim.x; i++) {
            for (size_t j = 0; j < dim.y; j++) {
                eulerian_v_h(i, j) = eulerian_v_raw[i + j * dim.x];
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
        double4 pos   = quadrature_rules(gidx);
        double  x     = (pos.x + 0.5 * h.x) / h.x;
        double  y     = pos.y / h.y;
        int     i     = floor(x);
        int     j     = floor(y);
        double  r1    = x - i;
        double  r2    = y - j;
        double  w[16] = {};
        double  v[16] = {};

        delta_function_weights(w, r1, r2);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) {
                    printf("出边界了！\n");
                    continue;
                }
                v[4 * (jj + 1) + (ii + 1)] = eulerian_v(ii + i, jj + j);
            }
        }

        double sum{};
        for (size_t i = 0; i < 16; i++)
            sum += v[i] * w[i];

        lagrange_v(gidx) = sum;
    }
};

struct LocalDistribute_u {
    typedef Kokkos::View<double*>  ViewVectorType;
    typedef Kokkos::View<double**> ViewMatrixType;
    typedef Kokkos::View<double4*> ViewDouble4Type;

    Kokkos::View<double*>  lagrange_u;
    Kokkos::View<double**> eulerian_u;
    Kokkos::View<double4*> quadrature_rules;

    // These variables should be moved to the GPU?
    double2 h;
    int2    dim;
    int     num;

    LocalDistribute_u(const double* lagrange_u_raw, const double4* quadrature_rules_raw, double2 h, int2 dim, int num)
        : lagrange_u("lagrange_u", num), eulerian_u("eulerian_u", dim.x, dim.y),
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
            for (size_t j = 0; j < dim.y; j++) {
                result[i + j * dim.x] = eulerian_u_h(i, j);
            }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos    = quadrature_rules(gidx);
        double  x      = pos.x / h.x;
        double  y      = (pos.y + 0.5 * h.y) / h.y;
        int     i      = floor(x);
        int     j      = floor(y);
        double  r1     = x - i;
        double  r2     = y - j;
        double  F      = lagrange_u[gidx];
        double  inv_h2 = 1.0 / h.x / h.y;
        double  w[16]  = {};
        double  f[16]  = {};

        delta_function_weights(w, r1, r2);

        for (size_t i = 0; i < 16; i++) {
            f[i] = inv_h2 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) { continue; }
                Kokkos::atomic_add(&eulerian_u(ii + i, jj + j), f[4 * (jj + 1) + (ii + 1)]);
            }
        }
    }
};

struct LocalDistribute_v {
    typedef Kokkos::View<double*>  ViewVectorType;
    typedef Kokkos::View<double**> ViewMatrixType;
    typedef Kokkos::View<double4*> ViewDouble4Type;

    Kokkos::View<double*>  lagrange_v;
    Kokkos::View<double**> eulerian_v;
    Kokkos::View<double4*> quadrature_rules;

    // These variables should be moved to the GPU?
    double2 h;
    int2    dim;
    int     num;

    LocalDistribute_v(const double* lagrange_v_raw, const double4* quadrature_rules_raw, double2 h, int2 dim, int num)
        : lagrange_v("lagrange_u", num), eulerian_v("eulerian_u", dim.x, dim.y),
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
            for (size_t j = 0; j < dim.y; j++) {
                result[i + j * dim.x] = eulerian_v_h(i, j);
            }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& gidx) const {
        double4 pos    = quadrature_rules(gidx);
        double  x      = (pos.x + 0.5 * h.x) / h.x;
        double  y      = pos.y / h.y;
        int     i      = floor(x);
        int     j      = floor(y);
        double  r1     = x - i;
        double  r2     = y - j;
        double  F      = lagrange_v[gidx];
        double  inv_h2 = 1.0 / h.x / h.y;
        double  w[16]  = {};
        double  f[16]  = {};

        delta_function_weights(w, r1, r2);

        for (size_t i = 0; i < 16; i++) {
            f[i] = inv_h2 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) { continue; }
                Kokkos::atomic_add(&eulerian_v(ii + i, jj + j), f[4 * (jj + 1) + (ii + 1)]);
            }
        }
    }
};

} // namespace mykokkos

void interpolate_u(double* lagrange_u, const double* eulerian_u, const double4* quadrature_rules, double2 h, int2 dim,
                   int num) {
    LOG_SCOPE_FUNCTION(INFO);

    // 初始化内核
    mykokkos::LocalInterpolate_u local_interpolate(eulerian_u, quadrature_rules, h, dim, num);

    // 并行计算
    Kokkos::parallel_for(local_interpolate.num, local_interpolate);

    // Copy data ro lagrange_u
    local_interpolate.extract_result(lagrange_u);
}

void interpolate_v(double* lagrange_v, const double* eulerian_v, const double4* quadrature_rules, double2 h, int2 dim,
                   int num) {
    LOG_SCOPE_FUNCTION(INFO);

    // 初始化内核
    mykokkos::LocalInterpolate_v local_interpolate(eulerian_v, quadrature_rules, h, dim, num);

    // 并行计算
    Kokkos::parallel_for(local_interpolate.num, local_interpolate);

    // Copy data ro lagrange_v
    local_interpolate.extract_result(lagrange_v);
}

void distribute_u(double* eulerian_u, const double* lagrange_u, const double4* quadrature_rules, double2 h, int2 dim,
                  int num) {
    LOG_SCOPE_FUNCTION(INFO);

    // 初始化内核
    mykokkos::LocalDistribute_u local_distribute(lagrange_u, quadrature_rules, h, dim, num);

    // 并行计算
    Kokkos::parallel_for(local_distribute.num, local_distribute);

    // Copy data ro lagrange_v
    local_distribute.extract_result(eulerian_u);
}

void distribute_v(double* eulerian_v, const double* lagrange_v, const double4* quadrature_rules, double2 h, int2 dim,
                  int num) {
    LOG_SCOPE_FUNCTION(INFO);

    // 初始化内核
    mykokkos::LocalDistribute_v local_distribute(lagrange_v, quadrature_rules, h, dim, num);

    // 并行计算
    Kokkos::parallel_for(local_distribute.num, local_distribute);

    // Copy data ro lagrange_v
    local_distribute.extract_result(eulerian_v);
}