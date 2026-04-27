/// @date 2023-12-19
/// @file VelocityU3D_kokkos.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#pragma once
#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra_kokkos.h>
#include <Kokkos_Core.hpp>
#include <io.h>

#include "../some_headers.h"

namespace mykokkos {

struct VelocityVKokkos3D {
    int    Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double dx, dy, dz, dt;
    double mu, rho, tolerance;
    int    max_iters;

    // xh : 速度u xb : 右端项 xr : 残差
    // xbcv : 边界值 xbct : 边界类型
    View3D<double> xh, xb, xr;
    View3D<double> x_prime;
    View3D<double> r_prime;
    View3D<double> xbcv;
    View3D<int>    xbct;

    VelocityVKokkos3D(int Nx, int Ny, int Nz, double dt, double Lx, double Ly, double Lz, double mu, double rho,
                      double tolerance, int max_iters)
        : Nx(Nx), Ny(Ny), Nz(Nz), Lx(Lx), Ly(Ly), Lz(Lz), dx(Lx / Nx), dy(Ly / Ny), dz(Lz / Nz), dt(dt), mu(mu),
          rho(rho), tolerance(tolerance), max_iters(max_iters), xh("xh", Nx + 2, Ny + 3, Nz + 2),
          xb("xb", Nx + 2, Ny + 3, Nz + 2), xr("xr", Nx + 2, Ny + 3, Nz + 2),
          x_prime("x_prime", Nx + 2, Ny + 3, Nz + 2), r_prime("r_prime", Nx + 2, Ny + 3, Nz + 2),
          xbcv("xbcv", Nx + 2, Ny + 3, Nz + 2), xbct("xbct", Nx + 2, Ny + 3, Nz + 2) {
        LOG_F(INFO, "Allocate memory for xh, xb, xr, x_prime, r_prime, xbcv, xbct");
    }

    template <int Surface>
    struct GenerateXPrime {
        View3D<double> x_prime, xbcv;
        View3D<int>    xbct;
        int            Nx, Ny, Nz;
        double         dx, dy, dz;
        GenerateXPrime(View3D<double> x_prime, View3D<double> xbcv, View3D<int> xbct, int Nx, int Ny, int Nz, double dx,
                       double dy, double dz)
            : x_prime(x_prime), xbcv(xbcv), xbct(xbct), Nx(Nx), Ny(Ny), Nz(Nz), dx(dx), dy(dy), dz(dz) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
            int x, y, z;
            if (Surface == LEFT) { x = 0, y = i, z = j + 1; }
            if (Surface == RIGHT) { x = Nx + 1, y = i, z = j + 1; }
            if (Surface == DOWN) { x = i + 1, y = 1, z = j + 1; }
            if (Surface == UP) { x = i + 1, y = Ny + 1, z = j + 1; }
            if (Surface == FRONT) { x = i + 1, y = j, z = 0; }
            if (Surface == BACK) { x = i + 1, y = j, z = Nz + 1; }

            if (Surface == LEFT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 2.0 * xbcv(x, y, z) - x_prime(x + 1, y, z); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = -dx * xbcv(x, y, z) + x_prime(x + 1, y, z); }
            }
            if (Surface == RIGHT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 2.0 * xbcv(x, y, z) - x_prime(x - 1, y, z); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = dx * xbcv(x, y, z) + x_prime(x - 1, y, z); }
            }

            if (Surface == DOWN) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = xbcv(x, y, z); }
                if (xbct(x, y, z) == NEUMANN) {
                    x_prime(x, y - 1, z) = -2.0 * dy * xbcv(x, y, z) + x_prime(x, y + 1, z);
                }
            }
            if (Surface == UP) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = xbcv(x, y, z); }
                if (xbct(x, y, z) == NEUMANN) {
                    x_prime(x, y + 1, z) = 2.0 * dy * xbcv(x, y, z) + x_prime(x, y - 1, z);
                }
            }

            if (Surface == FRONT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 2 * xbcv(x, y, z) - x_prime(x, y, z + 1); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = -dz * xbcv(x, y, z) + x_prime(x, y, z + 1); }
            }
            if (Surface == BACK) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 2 * xbcv(x, y, z) - x_prime(x, y, z - 1); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = dz * xbcv(x, y, z) + x_prime(x, y, z - 1); }
            }
        }
    };

    void generate_x_prime() {
        // HACK: 需要将 x_prime 设为零吗？

        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        GenerateXPrime<LEFT>  gxp_left(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        GenerateXPrime<RIGHT> gxp_right(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        GenerateXPrime<DOWN>  gxp_down(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        GenerateXPrime<UP>    gxp_up(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        GenerateXPrime<FRONT> gxp_front(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        GenerateXPrime<BACK>  gxp_back(x_prime, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);

        Kokkos::parallel_for("GXP_Left", policy_t({1, 0}, {Ny + 2, Nz}), gxp_left);
        Kokkos::parallel_for("GXP_right", policy_t({1, 0}, {Ny + 2, Nz}), gxp_right);
        Kokkos::parallel_for("GXP_down", policy_t({0, 0}, {Nx, Nz}), gxp_down);
        Kokkos::parallel_for("GXP_up", policy_t({0, 0}, {Nx, Nz}), gxp_up);
        Kokkos::parallel_for("GXP_front", policy_t({0, 1}, {Nx, Ny + 2}), gxp_front);
        Kokkos::parallel_for("GXP_back", policy_t({0, 1}, {Nx, Ny + 2}), gxp_back);
    }

    template <int Surface>
    struct SmoothBoundary {
        View3D<double> x_prime, xbcv;
        View3D<int>    xbct;
        int            Nx, Ny, Nz;
        double         dx, dy, dz;
        SmoothBoundary(View3D<double> x_prime, View3D<double> xbcv, View3D<int> xbct, int Nx, int Ny, int Nz, double dx,
                       double dy, double dz)
            : x_prime(x_prime), xbcv(xbcv), xbct(xbct), Nx(Nx), Ny(Ny), Nz(Nz), dx(dx), dy(dy), dz(dz) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
            int x, y, z;
            if (Surface == LEFT) { x = 0, y = i, z = j + 1; }
            if (Surface == RIGHT) { x = Nx + 1, y = i, z = j + 1; }
            if (Surface == DOWN) { x = i + 1, y = 1, z = j + 1; }
            if (Surface == UP) { x = i + 1, y = Ny + 1, z = j + 1; }
            if (Surface == FRONT) { x = i + 1, y = j, z = 0; }
            if (Surface == BACK) { x = i + 1, y = j, z = Nz + 1; }

            if (Surface == LEFT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = -x_prime(x + 1, y, z); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = x_prime(x + 1, y, z); }
            }
            if (Surface == RIGHT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = -x_prime(x - 1, y, z); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = x_prime(x - 1, y, z); }
            }

            if (Surface == DOWN) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 0.0; }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y - 1, z) = x_prime(x, y + 1, z); }
            }
            if (Surface == UP) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = 0.0; }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y + 1, z) = x_prime(x, y - 1, z); }
            }

            if (Surface == FRONT) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = -x_prime(x, y, z + 1); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = x_prime(x, y, z + 1); }
            }
            if (Surface == BACK) {
                if (xbct(x, y, z) == DIRICHLET) { x_prime(x, y, z) = -x_prime(x, y, z - 1); }
                if (xbct(x, y, z) == NEUMANN) { x_prime(x, y, z) = x_prime(x, y, z - 1); }
            }
        }
    };

    void smooth_boundary() {
        SmoothBoundary<LEFT>  smooth_boundary_left(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        SmoothBoundary<RIGHT> smooth_boundary_right(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        SmoothBoundary<DOWN>  smooth_boundary_down(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        SmoothBoundary<UP>    smooth_boundary_up(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        SmoothBoundary<FRONT> smooth_boundary_front(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        SmoothBoundary<BACK>  smooth_boundary_back(xh, xbcv, xbct, Nx, Ny, Nz, dx, dy, dz);
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        Kokkos::parallel_for("SmoothBoundary_Left", policy_t({1, 0}, {Ny + 2, Nz}), smooth_boundary_left);
        Kokkos::parallel_for("SmoothBoundary_Right", policy_t({1, 0}, {Ny + 2, Nz}), smooth_boundary_right);
        Kokkos::parallel_for("SmoothBoundary_Down", policy_t({0, 0}, {Nx, Nz}), smooth_boundary_down);
        Kokkos::parallel_for("SmoothBoundary_Up", policy_t({0, 0}, {Nx, Nz}), smooth_boundary_up);
        Kokkos::parallel_for("SmoothBoundary_front", policy_t({0, 1}, {Nx, Ny + 2}), smooth_boundary_front);
        Kokkos::parallel_for("SmoothBoundary_back", policy_t({0, 1}, {Nx, Ny + 2}), smooth_boundary_back);
    }

    struct Residual {
        View3D<double> xr, xh, xb;
        View3D<int>    xbct;
        int            Nz;
        double         dx, dy, dz, dt;
        double         mu, rho;

        Residual(View3D<double> xr, View3D<double> xh, View3D<double> xb, View3D<int> xbct, int Nz, double dx,
                 double dy, double dz, double dt, double mu, double rho)
            : xr(xr), xh(xh), xb(xb), xbct(xbct), Nz(Nz), dx(dx), dy(dy), dz(dz), dt(dt), mu(mu), rho(rho) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
            for (int k = 0; k < Nz; ++k) {
                if (xbct(i + 1, j, k + 1) == DIRICHLET) continue;
                double b1           = (xh(i, j, k + 1) + xh(i + 2, j, k + 1)) / (dx * dx);
                double b2           = (xh(i + 1, j - 1, k + 1) + xh(i + 1, j + 1, k + 1)) / (dy * dy);
                double b3           = (xh(i + 1, j, k) + xh(i + 1, j, k + 2)) / (dz * dz);
                double b            = mu * (b1 + b2 + b3);
                double a            = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                xr(i + 1, j, k + 1) = xb(i + 1, j, k + 1) + b - a * xh(i + 1, j, k + 1);
            }
        }
    };
    [[deprecated("Use another.")]] void load(const Array3D<double>& xh_, const Array3D<double>& xb_,
                                             const Array3D<double>& xbcv_, const Array3D<int>& xbct_) {
        View3D_h<double> xh_h   = Kokkos::create_mirror_view(xh);
        View3D_h<double> xb_h   = Kokkos::create_mirror_view(xb);
        View3D_h<double> xbcv_h = Kokkos::create_mirror_view(xbcv);
        View3D_h<int>    xbct_h = Kokkos::create_mirror_view(xbct);

        for (size_t i = 0; i < Nx + 2; i++)
            for (size_t j = 0; j < Ny + 3; j++)
                for (size_t k = 0; k < Nz + 2; k++) {
                    xh_h(i, j, k)   = xh_[i][j][k];
                    xb_h(i, j, k)   = xb_[i][j][k];
                    xbcv_h(i, j, k) = xbcv_[i][j][k];
                    xbct_h(i, j, k) = xbct_[i][j][k];
                }

        Kokkos::deep_copy(xh, xh_h);
        Kokkos::deep_copy(xb, xb_h);
        Kokkos::deep_copy(xbcv, xbcv_h);
        Kokkos::deep_copy(xbct, xbct_h);
    }
    void extract(Array3D<double>& xh_, Array3D<double>& xr_) {
        auto xh_h      = Kokkos::create_mirror_view(xh);
        auto xr_h      = Kokkos::create_mirror_view(xr);
        auto x_prime_h = Kokkos::create_mirror_view(x_prime);
        auto r_prime_h = Kokkos::create_mirror_view(r_prime);
        Kokkos::deep_copy(xh_h, xh);
        Kokkos::deep_copy(xr_h, xr);
        Kokkos::deep_copy(x_prime_h, x_prime);
        Kokkos::deep_copy(r_prime_h, r_prime);

        for (size_t i = 0; i < Nx + 2; i++)
            for (size_t j = 0; j < Ny + 3; j++)
                for (size_t k = 0; k < Nz + 2; k++) {
                    xh_[i][j][k] = xh_h(i, j, k);
                    xr_[i][j][k] = xr_h(i, j, k);
                    // x_prime_[i][j][k] = x_prime_h(i, j, k);
                    // r_prime_[i][j][k] = r_prime_h(i, j, k);
                }
    }

    void load(View3D<double> xh_, View3D<double> xb_, View3D<double> xbcv_, View3D<int> xbct_) {
        xh = xh_, xb = xb_, xbcv = xbcv_, xbct = xbct_;
    }

    template <typename L>
    auto convert(const Array3D<L>& data_, std::string name = "data") {
        // TODO: check size
        // 1. data_ 的每一个维度size()大于0
        // 2. data_ 的每一个成员的size()相等
        int Nx = data_.size();
        int Ny = data_[0].size();
        int Nz = data_[0][0].size();
        LOG_F(INFO, "convert: Nx = %d, Ny = %d, Nz = %d", Nx, Ny, Nz);
        View3D<L> data(name, Nx, Ny, Nz);

        View3D_h<L> data_h = Kokkos::create_mirror_view(data);
        for (size_t i = 0; i < Nx; i++)
            for (size_t j = 0; j < Ny; j++)
                for (size_t k = 0; k < Nz; k++) {
                    data_h(i, j, k) = data_[i][j][k];
                }
        Kokkos::deep_copy(data, data_h);

        return data;
    }

    void generate_r_prime() {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Residual grp(r_prime, x_prime, xb, xbct, Nz, dx, dy, dz, dt, mu, rho);

        Kokkos::parallel_for("GenerateRPrime", policy_t({0, 1}, {Nx, Ny + 2}), grp);
    }

    void residual() {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Residual grp(xr, xh, r_prime, xbct, Nz, dx, dy, dz, dt, mu, rho);

        Kokkos::parallel_for("Residual", policy_t({0, 1}, {Nx, Ny + 2}), grp);
    }

    template <int color, typename T>
    struct SmoothInner {
        View3D<T>   xh, xb;
        View3D<int> xbct;
        int         Nz;
        T           dx, dy, dz, dt;
        T           mu, rho;

        SmoothInner(View3D<T> xh, View3D<T> xb, View3D<int> xbct, int Nz, T dx, T dy, T dz, T dt, T rho, T mu)
            : xh(xh), xb(xb), xbct(xbct), Nz(Nz), dx(dx), dy(dy), dz(dz), dt(dt), rho(rho), mu(mu) {}
        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
            for (int k = 0; k < Nz; k++) {
                if (xbct(i + 1, j, k + 1) == DIRICHLET) continue;
                if ((i + j + k) % 2 == color) {
                    T b1                = (xh(i, j, k + 1) + xh(i + 2, j, k + 1)) / (dx * dx);
                    T b2                = (xh(i + 1, j - 1, k + 1) + xh(i + 1, j + 1, k + 1)) / (dy * dy);
                    T b3                = (xh(i + 1, j, k) + xh(i + 1, j, k + 2)) / (dz * dz);
                    T b                 = mu * (b1 + b2 + b3);
                    T a                 = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                    xh(i + 1, j, k + 1) = (xb(i + 1, j, k + 1) + b) / a;
                }
            }
        }
    };

    void smooth_inner(View3D<double>& xh, const View3D<int>& bct, const View3D<double>& xb, int3 N, double3 L) {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        // LOG_F(INFO, "VelocityU3D_kokkos::smooth, %d, %d, %d", N.x, N.y, N.z);
        SmoothInner<0, double> smooth_inner_0(xh, xb, bct, N.z, L.x / N.x, L.y / N.y, L.z / N.z, dt, rho, mu);
        SmoothInner<1, double> smooth_inner_1(xh, xb, bct, N.z, L.x / N.x, L.y / N.y, L.z / N.z, dt, rho, mu);
        Kokkos::parallel_for("SmoothInner0", policy_t({0, 1}, {N.x, N.y + 2}), smooth_inner_0);
        Kokkos::parallel_for("SmoothInner1", policy_t({0, 1}, {N.x, N.y + 2}), smooth_inner_1);
    }
};

// void inline solve_v(VelocityVKokkos3D& kokkos_solver, Array3D<double>& rh,
// Array3D<double>& xh,
//                     Array3D<double>& x_prime, Array3D<double>& r_prime,
//                     const MultiArrayInt& xbt, const Array3D<double>& xbv,
//                     const Array3D<double>& xb, int3 N, double3 L, double
//                     mu, double rho, double dt, int max_iters, double
//                     tolerance) {
//     double3 dh = {L.x / N.x, L.y / N.y, L.z / N.z};
//     kokkos_solver.load(xh, xb, xbv, xbt);
//     kokkos_solver.generate_x_prime();
//     kokkos_solver.generate_r_prime();
//     for (int iter = 0; iter < max_iters; iter++) {
//         kokkos_solver.smooth_boundary();
//         kokkos_solver.smooth_inner(kokkos_solver.xh, kokkos_solver.xbct,
//         kokkos_solver.r_prime, N, L);

//         if (iter % 1 == 0) {
//             kokkos_solver.smooth_boundary();
//             kokkos_solver.residual();

//             auto sum_r_squared = algebra::norm(kokkos_solver.xr) *
//             std::sqrt(dh.x * dh.y * dh.z); if (sum_r_squared < tolerance) {
//             break; }
//         }
//     }
//     algebra::add(kokkos_solver.xh, kokkos_solver.x_prime);
//     kokkos_solver.extract(xh, rh, x_prime, r_prime);
// }

} // namespace mykokkos