/// @date 2023-12-22
/// @file multigrid_components_U.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#include <PhysicsSolver/StokesFlow3D/Kokkos/Pressure3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityU3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityV3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityW3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridP.h>
#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridU.h>
#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridV.h>
#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridW.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityU3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityV3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityW3D.h>

// std::array<int, 6> all_boundary_type = {NEUMANN, NEUMANN, NEUMANN, NEUMANN,
// NEUMANN, NEUMANN};
std::array<int, 6> all_boundary_type = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
int                max_iters         = 20000;
double             tolerance         = 1e-6;

int main_v(int3 N, int Nt, double3 L, double T, double rho, double mu_f) {
    double t  = 0.0;
    double dx = L.x / N.x;
    double dy = L.y / N.y;
    double dz = L.z / N.z;
    double dt = T / Nt;

    auto ns_demo = create_ns_demo(N.x, N.y, N.z, Nt, L.x, L.y, L.z, T, rho, mu_f,
                                  "/home/kokkos/npuheart/features/finite_difference/3D/stokes_demo.json");

    auto xbt     = algebra::create_multi_array<3, int>({N.x + 2, N.y + 3, N.z + 2});
    auto xn      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto xb      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto xbv     = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto f2      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto exact   = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto eh      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto xh      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto rh      = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto x_prime = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});
    auto r_prime = algebra::create_multi_array<3, double>({N.x + 2, N.y + 3, N.z + 2});

    ns_demo->get_boundary_type_v(xbt, all_boundary_type);
    ns_demo->get_array_v(xn, t);

    auto kokkos_solver
        = mykokkos::VelocityVKokkos3D(N.x, N.y, N.z, Nt, L.x, L.y, L.z, T, mu_f, rho, tolerance, max_iters);
    auto view_xh  = kokkos_solver.template convert<double>(xh, "xh");
    auto view_xb  = kokkos_solver.template convert<double>(xb, "xb");
    auto view_xbt = kokkos_solver.template convert<int>(xbt, "xbt");
    auto view_xbv = kokkos_solver.template convert<double>(xbv, "xbv");

    for (int i = 1; i <= Nt; i++) {
        t = i * dt;
        // 获取边界条件和右端项
        ns_demo->get_array_f2(f2, t);
        ns_demo->make_tentitive_vb(xb, xn, f2);
        ns_demo->get_boundary_values_v(xbv, xbt, t);
        // 求解

        mykokkos::solve(kokkos_solver, rh, xh, xbt, xbv, xb, N, L, mu_f, rho, dt, max_iters, tolerance);
        // // HACK: 考虑到Kokkos::View默认是浅拷贝
        // 更新un
        // xn = xh; xh = xh;
        // std::swap(xn,xh);
        xn = xh;

        // 计算误差
        ns_demo->get_array_v(exact, t);
        algebra::axpy(-1.0, exact, xn, eh);
        algebra::zero_boundary(eh);
        double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);
        if (i == Nt) printf("sum_eh_squared = %.20e\n", sum_eh_squared);
    }

    return 0;
}

void main_test(int N_, int Nt) {
    int3    N    = {N_, N_, N_};
    double3 L    = {1.0, 1.0, 1.0};
    double  T    = 1.0;
    double  rho  = 1.0;
    double  mu_f = 1.0;
    main_v(N, Nt, L, T, rho, mu_f);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    std::vector<int> Ns{4};
    std::vector<int> Nt{10};
    for (auto& ns : Ns) {
        std::string msg = fmt::format("Result for ns = {}:", ns);
        std::cout << msg << std::endl;
        for (auto& nt : Nt) {
            std::string msg = fmt::format("           nt = {}:", nt);
            std::cout << msg << std::endl;
            main_test(ns, nt);
        }
    }
    Kokkos::finalize();
    return 0;
}