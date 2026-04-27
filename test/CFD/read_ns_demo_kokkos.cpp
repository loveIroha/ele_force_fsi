/// @date 2023-09-28
/// @file NavierStokesDemo.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 从Json文件读入三维NS方程解析解，并生成可计算对象。
///
///

#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/VelocityU3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityV3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityW3D.h>

int max_iters = 200000;
// std::array<int, 6> all_boundary_type = {2, 2, 2, 2, 2, 2};
std::array<int, 6> all_boundary_type = {2, 2, 2, 2, 2, 2};
double             tolerance         = 1e-6;

void solve_p(MultiArrayDouble& ph, MultiArrayDouble& r, const MultiArrayInt& pbt, const MultiArrayDouble& pbv,
             const MultiArrayDouble& bp, int Nx, int Ny, int Nz, double width, double height, double depth,
             int max_iters, double tolerance) {
    LOG_SCOPE_FUNCTION(INFO);
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    auto kokkos_solver
        = mykokkos::PressureKokkos3D(Nx, Ny, Nz, 1, width, height, depth, 1, 0.1, 1.0, tolerance, max_iters);

    auto view_ph  = kokkos_solver.template convert<double>(ph, "ph");
    auto view_pb  = kokkos_solver.template convert<double>(bp, "pb");
    auto view_pbt = kokkos_solver.template convert<int>(pbt, "pbt");
    auto view_pbv = kokkos_solver.template convert<double>(pbv, "pbv");

    kokkos_solver.load(view_ph, view_pb, view_pbv, view_pbt);

    kokkos_solver.generate_p_prime();
    kokkos_solver.generate_r_prime();

    auto p_prime = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    auto r_prime = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    kokkos_solver.extract(ph, r, p_prime, r_prime);

    for (int iter = 0; iter < 1000; ++iter) {
        kokkos_solver.smooth_boundary();
        kokkos_solver.smooth_inner();
        kokkos_solver.smooth_boundary();
        kokkos_solver.residual();
        double sum_r_squared = kokkos_solver.residual_norm();

        // pressure_3D::smooth(ph, r_prime, pbt, Nx, Ny, Nz, width, height,
        // depth); pressure_3D::residual(r, ph, r_prime, pbt, Nx, Ny, Nz, width,
        // height, depth); double sum_r_squared = algebra::norm(r) * std::sqrt(dx
        // * dy * dz);

        // Check convergence condition
        LOG_F(INFO, "iter = %d, sum_r_squared = %.20e", iter, sum_r_squared);
        // LOG_F(INFO, "iter = %d.", iter);
        if (sum_r_squared < tolerance) { break; }
    }
    kokkos_solver.extract(ph, r, p_prime, r_prime);
}

int main_p(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
           double mu_f) {
    auto ns_demo = create_ns_demo(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);

    double t  = 0.0;
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;
    // double dt = T / Nt;

    // 获取边界条件和右端项
    auto pbt = algebra::create_multi_array<3, int>({Nx + 2, Ny + 2, Nz + 2});
    auto bp  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    auto pbv = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});

    ns_demo->get_boundary_type_p(pbt, all_boundary_type);
    ns_demo->get_array_bp(bp, t);
    ns_demo->get_boundary_values_p(pbv, pbt, t);

    // 求解
    auto ph = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    auto r  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    solve_p(ph, r, pbt, pbv, bp, Nx, Ny, Nz, width, height, depth, max_iters, tolerance);

    // 计算误差
    MultiArrayDouble exact = ns_demo->get_array_p(t);
    auto             eh    = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    algebra::axpy(-1.0, exact, ph, eh);
    algebra::zero_boundary(eh);
    double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);
    printf("sum_eh_squared = %.20e\n", sum_eh_squared);

    std::ofstream out_pbt("pbt.txt");
    // std::ofstream out_bp("bp.txt");
    std::ofstream out_p("p_exact.txt");
    std::ofstream out_pbv("pbv.txt");
    algebra::print_multi_array<3, int>(pbt, out_pbt);
    // algebra::print_multi_array<3, double>(bp, out_bp);
    algebra::print_multi_array<3, double>(exact, out_p);
    // algebra::print_multi_array<3, double>(pbv, out_pbv);
    return 0;
}

void main_test(int N, int Nt) {
    int    Nx     = N;
    int    Ny     = N;
    int    Nz     = N;
    double width  = 1.0;
    double height = 1.0;
    double depth  = 1.0;
    double T      = 1.0;
    double rho    = 1.0;
    double mu_f   = 1.0;
    main_p(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
}

int main(int argc, char* argv[]) {
    auto settings = Kokkos::InitializationSettings().set_device_id(1);

    Kokkos::initialize(settings);

    std::vector<int> Ns{256};
    std::vector<int> Nt{1};

    for (auto& ns : Ns) {
        std::string msg = "Result for ns = {}:";
        std::cout << msg << std::endl;
        for (auto& nt : Nt) {
            std::string msg = "           nt = {}:";
            std::cout << msg << std::endl;
            main_test(ns, nt);
        }
    }
    Kokkos::finalize();
    return 0;
}