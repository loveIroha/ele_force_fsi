/// @date 2023-12-11
/// @file Pressure3D_kokkos.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>

#include "../some_headers.h"

namespace mykokkos {

struct PressureKokkos3D {
    bool PureNeumann = false;

    int    Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double dx, dy, dz, dt;
    double mu, rho, tolerance;
    int    max_iters;

    // ph : 压强 pb : 右端项 pr : 残差
    // pbcv : 边界值 pbct : 边界类型
    View3D<double> ph, pb, pr;
    View3D<double> p_prime;
    View3D<double> r_prime;
    View3D<double> pbcv;
    View3D<int>    pbct;

    PressureKokkos3D(int Nx, int Ny, int Nz, double dt, double Lx, double Ly, double Lz, double mu, double rho,
                     double tolerance, int max_iters)
        : Nx(Nx), Ny(Ny), Nz(Nz), Lx(Lx), Ly(Ly), Lz(Lz), dx(Lx / Nx), dy(Ly / Ny), dz(Lz / Nz),
          dt(dt), max_iters(max_iters), tolerance(tolerance), ph("ph", Nx + 2, Ny + 2, Nz + 2),
          pb("pb", Nx + 2, Ny + 2, Nz + 2), pr("pr", Nx + 2, Ny + 2, Nz + 2),
          p_prime("p_prime", Nx + 2, Ny + 2, Nz + 2), r_prime("r_prime", Nx + 2, Ny + 2, Nz + 2),
          pbcv("pbcv", Nx + 2, Ny + 2, Nz + 2), pbct("pbct", Nx + 2, Ny + 2, Nz + 2) {}

    // 内点循环
    template <int color, typename T>
    class SmoothInner {}; // WorkTag for Kokkos
    template <int color, typename T>
    KOKKOS_FUNCTION void operator()(SmoothInner<color, T>, int i, int j) const {
        for (int k = 0; k < Nz; k++) {
            if ((i + j + k) % 2 == color) {
                T b1 = (ph(i + 2, j + 1, k + 1) + ph(i, j + 1, k + 1)) / (dx * dx);
                T b2 = (ph(i + 1, j + 2, k + 1) + ph(i + 1, j, k + 1)) / (dy * dy);
                T b3 = (ph(i + 1, j + 1, k + 2) + ph(i + 1, j + 1, k)) / (dz * dz);
                T a  = -2 / (dx * dx) - 2 / (dy * dy) - 2 / (dz * dz);
                T b  = b1 + b2 + b3;

                ph(i + 1, j + 1, k + 1) = (r_prime(i + 1, j + 1, k + 1) - b) / a;
            }
        }
    }

    void smooth_inner() {
        using policy_red_t   = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothInner<0, double>>;
        using policy_black_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothInner<1, double>>;
        Kokkos::parallel_for("SmoothInnerRed", policy_red_t({0, 0}, {Nx, Ny}), *this);
        Kokkos::parallel_for("SmoothInnerBlack", policy_black_t({0, 0}, {Nx, Ny}), *this);
    };

    // 边界点循环
    template <int Surface>
    class SmoothBoundary {};

    template <int Surface>
    KOKKOS_FUNCTION void smooth_boundary(const View3D<double> ph, int i, int j) const {
        int x, y, z;
        if (Surface == LEFT) { x = 0, y = i + 1, z = j + 1; }
        if (Surface == RIGHT) { x = Nx + 1, y = i + 1, z = j + 1; }
        if (Surface == DOWN) { x = i + 1, y = 0, z = j + 1; }
        if (Surface == UP) { x = i + 1, y = Ny + 1, z = j + 1; }
        if (Surface == FRONT) { x = i + 1, y = j + 1, z = 0; }
        if (Surface == BACK) { x = i + 1, y = j + 1, z = Nz + 1; }
        if (Surface == LEFT) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x + 1, y, z); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x + 1, y, z); }
        }
        if (Surface == RIGHT) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x - 1, y, z); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x - 1, y, z); }
        }
        if (Surface == DOWN) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x, y + 1, z); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x, y + 1, z); }
        }
        if (Surface == UP) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x, y - 1, z); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x, y - 1, z); }
        }
        if (Surface == FRONT) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x, y, z + 1); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x, y, z + 1); }
        }
        if (Surface == BACK) {
            if (pbct(x, y, z) == DIRICHLET) { ph(x, y, z) = -ph(x, y, z - 1); }
            if (pbct(x, y, z) == NEUMANN) { ph(x, y, z) = ph(x, y, z - 1); }
        }
    }
    template <int Surface>
    KOKKOS_FUNCTION void operator()(SmoothBoundary<Surface>, int i, int j) const {
        smooth_boundary<Surface>(ph, i, j);
    }
    void smooth_boundary() {
        using policy_left_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<LEFT>>;
        using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<RIGHT>>;
        using policy_down_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<DOWN>>;
        using policy_up_t    = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<UP>>;
        using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<FRONT>>;
        using policy_back_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, SmoothBoundary<BACK>>;

        Kokkos::parallel_for("SmoothBoundary_Left", policy_left_t({0, 0}, {Ny, Nz}), *this);
        Kokkos::parallel_for("SmoothBoundary_Right", policy_right_t({0, 0}, {Ny, Nz}), *this);
        Kokkos::parallel_for("SmoothBoundary_Down", policy_down_t({0, 0}, {Nx, Nz}), *this);
        Kokkos::parallel_for("SmoothBoundary_Up", policy_up_t({0, 0}, {Nx, Nz}), *this);
        Kokkos::parallel_for("SmoothBoundary_front", policy_front_t({0, 0}, {Nx, Ny}), *this);
        Kokkos::parallel_for("SmoothBoundary_back", policy_back_t({0, 0}, {Nx, Ny}), *this);
    };

    // generate_p_prime
    template <int Surface>
    class GeneratePPrime {};
    template <int Surface>
    KOKKOS_FUNCTION void operator()(GeneratePPrime<Surface>, int i, int j) const {
        int x, y, z;
        if (Surface == LEFT) { x = 0, y = i + 1, z = j + 1; }
        if (Surface == RIGHT) { x = Nx + 1, y = i + 1, z = j + 1; }
        if (Surface == DOWN) { x = i + 1, y = 0, z = j + 1; }
        if (Surface == UP) { x = i + 1, y = Ny + 1, z = j + 1; }
        if (Surface == FRONT) { x = i + 1, y = j + 1, z = 0; }
        if (Surface == BACK) { x = i + 1, y = j + 1, z = Nz + 1; }

        if (Surface == LEFT) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x + 1, y, z); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = -dx * pbcv(x, y, z) + p_prime(x + 1, y, z); }
        }
        if (Surface == RIGHT) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x - 1, y, z); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = dx * pbcv(x, y, z) + p_prime(x - 1, y, z); }
        }

        if (Surface == DOWN) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x, y + 1, z); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = -dy * pbcv(x, y, z) + p_prime(x, y + 1, z); }
        }

        if (Surface == UP) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x, y - 1, z); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = dy * pbcv(x, y, z) + p_prime(x, y - 1, z); }
        }

        if (Surface == FRONT) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x, y, z + 1); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = -dz * pbcv(x, y, z) + p_prime(x, y, z + 1); }
        }
        if (Surface == BACK) {
            if (pbct(x, y, z) == DIRICHLET) { p_prime(x, y, z) = 2 * pbcv(x, y, z) - p_prime(x, y, z - 1); }
            if (pbct(x, y, z) == NEUMANN) { p_prime(x, y, z) = dz * pbcv(x, y, z) + p_prime(x, y, z - 1); }
        }
    }
    void generate_p_prime() {
        using policy_left_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<LEFT>>;
        using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<RIGHT>>;
        using policy_down_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<DOWN>>;
        using policy_up_t    = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<UP>>;
        using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<FRONT>>;
        using policy_back_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GeneratePPrime<BACK>>;

        Kokkos::parallel_for("GeneratePPrime_Left", policy_left_t({0, 0}, {Ny, Nz}), *this);
        Kokkos::parallel_for("GeneratePPrime_Right", policy_right_t({0, 0}, {Ny, Nz}), *this);
        Kokkos::parallel_for("GeneratePPrime_Down", policy_down_t({0, 0}, {Nx, Nz}), *this);
        Kokkos::parallel_for("GeneratePPrime_Up", policy_up_t({0, 0}, {Nx, Nz}), *this);
        Kokkos::parallel_for("GeneratePPrime_front", policy_front_t({0, 0}, {Nx, Ny}), *this);
        Kokkos::parallel_for("GeneratePPrime_back", policy_back_t({0, 0}, {Nx, Ny}), *this);
    }

    // generate_r_prime
    KOKKOS_FUNCTION void residual(const View3D<double> pr, const View3D<double> ph, const View3D<double> pb, int i,
                                  int j) const {
        for (int k = 0; k < Nz; k++) {
            double ra = (ph(i + 2, j + 1, k + 1) - 2 * ph(i + 1, j + 1, k + 1) + ph(i, j + 1, k + 1)) / (dx * dx);
            double rb = (ph(i + 1, j + 2, k + 1) - 2 * ph(i + 1, j + 1, k + 1) + ph(i + 1, j, k + 1)) / (dy * dy);
            double rc = (ph(i + 1, j + 1, k + 2) - 2 * ph(i + 1, j + 1, k + 1) + ph(i + 1, j + 1, k)) / (dz * dz);

            pr(i + 1, j + 1, k + 1) = pb(i + 1, j + 1, k + 1) - ra - rb - rc;
        }
    }
    class GenerateRPrime {};
    KOKKOS_FUNCTION void operator()(GenerateRPrime, int i, int j) const { residual(r_prime, p_prime, pb, i, j); }
    void                 generate_r_prime() {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, GenerateRPrime>;
        Kokkos::parallel_for("GenerateRPrime", policy_t({0, 0}, {Nx, Ny}), *this);
    }

    class Residual {};
    KOKKOS_FUNCTION void operator()(Residual, int i, int j) const { residual(pr, ph, r_prime, i, j); }

    void residual() {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Residual>;
        Kokkos::parallel_for("Residual", policy_t({0, 0}, {Nx, Ny}), *this);
    }

    // 残差的范数
    class ResidualNorm {};
    KOKKOS_FUNCTION void operator()(ResidualNorm, int i, int j, int k, double& norm) const {
        norm += pr(i + 1, j + 1, k + 1) * pr(i + 1, j + 1, k + 1);
    }
    double residual_norm() {
        double norm    = 0;
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, ResidualNorm>;
        Kokkos::parallel_reduce("ResidualNorm", policy_t({0, 0, 0}, {Nx, Ny, Nz}), *this, norm);
        return std::sqrt(norm) * std::sqrt(dx * dy * dz);
    }

    void load(View3D<double> ph_, View3D<double> pb_, View3D<double> pbcv_, View3D<int> pbct_) {
        ph = ph_, pb = pb_, pbcv = pbcv_, pbct = pbct_;
    }

    [[deprecated("Use another.")]] void load(const Array3D<double>& ph_, const Array3D<double>& pb_,
                                             const Array3D<double>& pbcv_, const Array3D<int>& pbct_) {
        View3D<double>::HostMirror ph_h   = Kokkos::create_mirror_view(ph);
        View3D<double>::HostMirror pb_h   = Kokkos::create_mirror_view(pb);
        View3D<double>::HostMirror pbcv_h = Kokkos::create_mirror_view(pbcv);
        View3D<int>::HostMirror    pbct_h = Kokkos::create_mirror_view(pbct);

        for (size_t i = 0; i < Nx + 2; i++)
            for (size_t j = 0; j < Ny + 2; j++)
                for (size_t k = 0; k < Nz + 2; k++) {
                    ph_h(i, j, k)   = ph_[i][j][k];
                    pb_h(i, j, k)   = pb_[i][j][k];
                    pbcv_h(i, j, k) = pbcv_[i][j][k];
                    pbct_h(i, j, k) = pbct_[i][j][k];
                }

        Kokkos::deep_copy(ph, ph_h);
        Kokkos::deep_copy(pb, pb_h);
        Kokkos::deep_copy(pbcv, pbcv_h);
        Kokkos::deep_copy(pbct, pbct_h);
    }
    void extract(Array3D<double>& ph_, Array3D<double>& pr_, Array3D<double>& p_prime_, Array3D<double>& r_prime_) {
        auto ph_h      = Kokkos::create_mirror_view(ph);
        auto pr_h      = Kokkos::create_mirror_view(pr);
        auto p_prime_h = Kokkos::create_mirror_view(p_prime);
        auto r_prime_h = Kokkos::create_mirror_view(r_prime);
        Kokkos::deep_copy(ph_h, ph);
        Kokkos::deep_copy(pr_h, pr);
        Kokkos::deep_copy(p_prime_h, p_prime);
        Kokkos::deep_copy(r_prime_h, r_prime);

        for (size_t i = 0; i < Nx + 2; i++)
            for (size_t j = 0; j < Ny + 2; j++)
                for (size_t k = 0; k < Nz + 2; k++) {
                    ph_[i][j][k]      = ph_h(i, j, k);
                    pr_[i][j][k]      = pr_h(i, j, k);
                    p_prime_[i][j][k] = p_prime_h(i, j, k);
                    r_prime_[i][j][k] = r_prime_h(i, j, k);
                }
    }
};

} // namespace mykokkos
