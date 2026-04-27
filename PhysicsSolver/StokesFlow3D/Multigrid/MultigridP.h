/// @date 2023-12-18
/// @file MultigridP.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei & Wang Xuan
///
/// @brief
///
///

#pragma once
#include "../Kokkos/Pressure3D_kokkos.h"
#include "Multigrid.h"

namespace mg {
class MultigridP : public Multigrid<double, View3D> {
  private:
  public:
    MultigridP(int3 N, double3 L, View3D<int> bct) : Multigrid{N, L, bct} {}

    virtual void restrict(View3D<double>& coarse, const View3D<double>& fine, int3 N) override final {
        // LOG_F(INFO, "MultigridP::restrict");
        typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
        Kokkos::parallel_for(
            "restrict", mdrange_policy({0, 0, 0}, {N.x, N.y, N.z}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                coarse(i + 1, j + 1, k + 1)
                    = (fine(2 * i + 1, 2 * j + 1, 2 * k + 2) + fine(2 * i + 2, 2 * j + 1, 2 * k + 2)
                       + fine(2 * i + 1, 2 * j + 2, 2 * k + 2) + fine(2 * i + 2, 2 * j + 2, 2 * k + 2)
                       + fine(2 * i + 1, 2 * j + 1, 2 * k + 1) + fine(2 * i + 2, 2 * j + 1, 2 * k + 1)
                       + fine(2 * i + 1, 2 * j + 2, 2 * k + 1) + fine(2 * i + 2, 2 * j + 2, 2 * k + 1))
                      / 8.0;
            });
    }

    virtual void restrict_bc(View3D<int>& coarse, const View3D<int>& fine, int3 N) override final {
        // LOG_F(INFO, "MultigridP::restrict_bc");
        typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;
        Kokkos::parallel_for(
            "restrict_bc_1", mdrange_policy({0, 0}, {N.y, N.z}), KOKKOS_LAMBDA(const int j, const int k) {
                coarse(0, j + 1, k + 1) = max(max(fine(0, 2 * j + 1, 2 * k + 1), fine(0, 2 * j + 2, 2 * k + 1)),
                                              max(fine(0, 2 * j + 1, 2 * k + 2), fine(0, 2 * j + 2, 2 * k + 2)));
                coarse(N.x + 1, j + 1, k + 1)
                    = max(max(fine(2 * N.x + 1, 2 * j + 1, 2 * k + 1), fine(2 * N.x + 1, 2 * j + 2, 2 * k + 1)),
                          max(fine(2 * N.x + 1, 2 * j + 1, 2 * k + 2), fine(2 * N.x + 1, 2 * j + 2, 2 * k + 2)));
            });
        Kokkos::parallel_for(
            "restrict_bc_2", mdrange_policy({0, 0}, {N.x, N.z}), KOKKOS_LAMBDA(const int i, const int k) {
                coarse(i + 1, 0, k + 1) = max(max(fine(2 * i + 1, 0, 2 * k + 1), fine(2 * i + 2, 0, 2 * k + 1)),
                                              max(fine(2 * i + 1, 0, 2 * k + 2), fine(2 * i + 2, 0, 2 * k + 2)));

                coarse(i + 1, N.y + 1, k + 1)
                    = max(max(fine(2 * i + 1, 2 * N.y + 1, 2 * k + 1), fine(2 * i + 2, 2 * N.y + 1, 2 * k + 1)),
                          max(fine(2 * i + 1, 2 * N.y + 1, 2 * k + 2), fine(2 * i + 2, 2 * N.y + 1, 2 * k + 2)));
            });
        Kokkos::parallel_for(
            "restrict_bc_3", mdrange_policy({0, 0}, {N.x, N.y}), KOKKOS_LAMBDA(const int i, const int j) {
                coarse(i + 1, j + 1, 0) = max(max(fine(2 * i + 1, 2 * j + 1, 0), fine(2 * i + 2, 2 * j + 1, 0)),
                                              max(fine(2 * i + 1, 2 * j + 2, 0), fine(2 * i + 2, 2 * j + 2, 0)));

                coarse(i + 1, j + 1, N.z + 1)
                    = max(max(fine(2 * i + 1, 2 * j + 1, 2 * N.z + 1), fine(2 * i + 2, 2 * j + 1, 2 * N.z + 1)),
                          max(fine(2 * i + 1, 2 * j + 2, 2 * N.z + 1), fine(2 * i + 2, 2 * j + 2, 2 * N.z + 1)));
            });
    }

    virtual void interpolate(View3D<double>& fine, const View3D<double>& coarse, int3 N) override {
        // LOG_F(INFO, "MultigridP::interpolate");
        typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
        Kokkos::parallel_for(
            "interpolate", mdrange_policy({0, 0, 0}, {N.x + 1, N.y + 1, N.z + 1}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                double f[8] = {coarse(i, j, k),         coarse(i + 1, j, k),        coarse(i, j + 1, k),
                               coarse(i + 1, j + 1, k), coarse(i, j, k + 1),        coarse(i + 1, j, k + 1),
                               coarse(i, j + 1, k + 1), coarse(i + 1, j + 1, k + 1)};

                fine(2 * i, 2 * j, 2 * k)             = trilinear(0.25, 0.25, 0.25, f);
                fine(2 * i, 2 * j, 2 * k + 1)         = trilinear(0.25, 0.25, 0.75, f);
                fine(2 * i, 2 * j + 1, 2 * k)         = trilinear(0.25, 0.75, 0.25, f);
                fine(2 * i, 2 * j + 1, 2 * k + 1)     = trilinear(0.25, 0.75, 0.75, f);
                fine(2 * i + 1, 2 * j, 2 * k)         = trilinear(0.75, 0.25, 0.25, f);
                fine(2 * i + 1, 2 * j, 2 * k + 1)     = trilinear(0.75, 0.25, 0.75, f);
                fine(2 * i + 1, 2 * j + 1, 2 * k)     = trilinear(0.75, 0.75, 0.25, f);
                fine(2 * i + 1, 2 * j + 1, 2 * k + 1) = trilinear(0.75, 0.75, 0.75, f);
            });
    }

    template <int color, typename T>
    struct SmoothInner {
        View3D<T> ph, r_prime;
        int       Nz;
        T         dx, dy, dz;
        SmoothInner(View3D<T> ph, View3D<T> r_prime, int Nz, T dx, T dy, T dz)
            : ph(ph), r_prime(r_prime), Nz(Nz), dx(dx), dy(dy), dz(dz) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
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
    };
    virtual void smooth(View3D<double>& p, const View3D<int>& bct, const View3D<double>& b, int3 N,
                        double3 L) override {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        SmoothInner<0, double> smooth_inner_red(p, b, N.z, L.x / N.x, L.y / N.y, L.z / N.z);
        SmoothInner<1, double> smooth_inner_black(p, b, N.z, L.x / N.x, L.y / N.y, L.z / N.z);
        Kokkos::parallel_for("SmoothInner0", policy_t({0, 0}, {N.x, N.y}), smooth_inner_red);
        Kokkos::parallel_for("SmoothInner1", policy_t({0, 0}, {N.x, N.y}), smooth_inner_black);
    }

    template <int Surface>
    struct SmoothBoundary {
        View3D<double> ph;
        View3D<int>    pbct;
        int            Nx, Ny, Nz;
        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
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
    };

    virtual void smooth_boundary(View3D<double>& ph, const View3D<int>& bct, int3 N) override {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        SmoothBoundary<LEFT>  smooth_boundary_left(ph, bct, N.x, N.y, N.z);
        SmoothBoundary<RIGHT> smooth_boundary_right(ph, bct, N.x, N.y, N.z);
        SmoothBoundary<DOWN>  smooth_boundary_down(ph, bct, N.x, N.y, N.z);
        SmoothBoundary<UP>    smooth_boundary_up(ph, bct, N.x, N.y, N.z);
        SmoothBoundary<FRONT> smooth_boundary_front(ph, bct, N.x, N.y, N.z);
        SmoothBoundary<BACK>  smooth_boundary_back(ph, bct, N.x, N.y, N.z);

        Kokkos::parallel_for("SmoothBoundary_Left", policy_t({0, 0}, {N.y, N.z}), smooth_boundary_left);
        Kokkos::parallel_for("SmoothBoundary_Right", policy_t({0, 0}, {N.y, N.z}), smooth_boundary_right);
        Kokkos::parallel_for("SmoothBoundary_Down", policy_t({0, 0}, {N.x, N.z}), smooth_boundary_down);
        Kokkos::parallel_for("SmoothBoundary_Up", policy_t({0, 0}, {N.x, N.z}), smooth_boundary_up);
        Kokkos::parallel_for("SmoothBoundary_front", policy_t({0, 0}, {N.x, N.y}), smooth_boundary_front);
        Kokkos::parallel_for("SmoothBoundary_back", policy_t({0, 0}, {N.x, N.y}), smooth_boundary_back);
    };

    template <typename T>
    struct Residual {
        View3D<T> pr, ph, pb;
        int       Nz;
        T         dx, dy, dz;
        Residual(View3D<T> pr, View3D<T> ph, View3D<T> pb, int Nz, T dx, T dy, T dz)
            : pr(pr), ph(ph), pb(pb), Nz(Nz), dx(dx), dy(dy), dz(dz) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(int i, int j) const {
            for (int k = 0; k < Nz; k++) {
                T ra = (ph(i + 2, j + 1, k + 1) - 2 * ph(i + 1, j + 1, k + 1) + ph(i, j + 1, k + 1)) / (dx * dx);
                T rb = (ph(i + 1, j + 2, k + 1) - 2 * ph(i + 1, j + 1, k + 1) + ph(i + 1, j, k + 1)) / (dy * dy);
                T rc = (ph(i + 1, j + 1, k + 2) - 2 * ph(i + 1, j + 1, k + 1) + ph(i + 1, j + 1, k)) / (dz * dz);

                pr(i + 1, j + 1, k + 1) = pb(i + 1, j + 1, k + 1) - ra - rb - rc;
            }
        }
    };

    virtual void compute_residual(View3D<double>& r, View3D<double>& p, const View3D<int>& bct, const View3D<double>& b,
                                  int3 N, double3 L) override {
        using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Residual<double> residual(r, p, b, N.z, L.x / N.x, L.y / N.y, L.z / N.z);
        Kokkos::parallel_for("residual", policy_t({0, 0}, {N.x, N.y}), residual);
    }

    ~MultigridP() override {}
};
} // namespace mg

namespace mykokkos {

template <>
inline void solve<PressureKokkos3D>(const std::unique_ptr<PressureKokkos3D>& kokkos_solver, Array3D<double>& rh,
                                    Array3D<double>& xh, const Array3D<int>& xbt, const Array3D<double>& xbv,
                                    const Array3D<double>& xb, int3 N, double3 L, double mu, double rho, double dt,
                                    int max_iters, double tolerance) {
    LOG_SCOPE_FUNCTION(WATCH);
    LOG_F(INFO, "多重网格迭代");

    auto view_xh  = to_View<double>(xh, "ph");
    auto view_xb  = to_View<double>(xb, "pb");
    auto view_xbt = to_View<int>(xbt, "pbt");
    auto view_xbv = to_View<double>(xbv, "pbv");

    auto mgsolver         = mg::MultigridFactory<double, mg::MultigridP, View3D>::createObject(N, L, view_xbt);
    mgsolver->PureNeumann = kokkos_solver->PureNeumann;

    // 转换为齐次问题 A*p_prime=r_prime
    kokkos_solver->load(view_xh, view_xb, view_xbv, view_xbt);
    kokkos_solver->generate_p_prime();
    kokkos_solver->generate_r_prime();

    // GS-RB迭代求解
    // for (size_t i = 0; i < 10000; i++) {
    //     kokkos_solver->smooth_boundary();
    //     kokkos_solver->smooth_inner();
    //     kokkos_solver->smooth_boundary();
    //     kokkos_solver->residual();
    //     auto a = kokkos_solver->residual_norm();
    //     printf("%d residual norm : %.20f\n",i,  a);
    // }

    // 多重网格求解 A*p_prime=r_prime
    mgsolver->vcycle(kokkos_solver->ph, kokkos_solver->r_prime);

    // kokkos_solver->ph += kokkos_solver->p_prime;
    mg::add(kokkos_solver->ph, kokkos_solver->p_prime);

    // Remove the average value of ph
    if (mgsolver->PureNeumann) {
        auto aver = algebra::sum(kokkos_solver->ph);
        LOG_F(INFO, "压强的边界条件为纯Neumann边界条件，p的平均值为%.20f", aver);
        algebra::add(kokkos_solver->ph, -aver);
        aver = algebra::sum(kokkos_solver->ph);
        LOG_F(INFO, "修改后，p的平均值为%.20f", aver);
    }
    // 提取出ph
    mgsolver->extract_x(xh, 0);
}
} // namespace mykokkos