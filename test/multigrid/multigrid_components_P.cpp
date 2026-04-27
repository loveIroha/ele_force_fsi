/// @date 2023-12-13
/// @file multigrid_components.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridP.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo.h>

namespace mykokkos {

void mg3D_p_solver(int N_, int Nt) {
    double  t         = 0.0;
    int3    N         = {N_, N_, N_};
    double3 L         = {1.0, 1.0, 1.0};
    double3 dh        = {L.x / N.x, L.y / N.y, L.z / N.z};
    double  Lt        = 1.0;
    double  rho       = 1.0;
    double  mu_f      = 0.1;
    double  tolerance = 1e-6;
    int     max_iters = 1000;
    // std::array<int, 6> all_boundary_type = {DIRICHLET, DIRICHLET, DIRICHLET,
    // DIRICHLET, DIRICHLET, DIRICHLET};
    std::array<int, 6> all_boundary_type = {NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN};
    auto               ns_demo           = create_ns_demo(N.x, N.y, N.z, Nt, L.x, L.y, L.z, Lt, rho, mu_f);
    auto kokkos_solver = std::make_unique<mykokkos::PressureKokkos3D>(N.x, N.y, N.z, Nt, L.x, L.y, L.z, Lt, mu_f, rho,
                                                                      tolerance, max_iters);

    {
        auto all_neumann = all_boundary_type[0] == NEUMANN && all_boundary_type[1] == NEUMANN
                           && all_boundary_type[2] == NEUMANN && all_boundary_type[3] == NEUMANN
                           && all_boundary_type[4] == NEUMANN && all_boundary_type[5] == NEUMANN;
        if (all_neumann) {
            kokkos_solver->PureNeumann = true;
            LOG_F(WARNING, "pure neumann boundary conditions are applied!");
        }
    }

    auto pbt = algebra::create_multi_array<3, int>({N.x + 2, N.y + 2, N.z + 2});
    auto ph  = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});
    auto p_  = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});
    auto pr  = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});
    auto pb  = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});
    auto pe  = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});
    auto pbv = algebra::create_multi_array<3, double>({N.x + 2, N.y + 2, N.z + 2});

    ns_demo->get_boundary_type_p(pbt, all_boundary_type);
    ns_demo->get_array_bp(pb, t);
    ns_demo->get_boundary_values_p(pbv, pbt, t);

    mykokkos::solve(kokkos_solver, pr, ph, pbt, pbv, pb, N, L, mu_f, rho, Lt / Nt, max_iters, tolerance);

    ns_demo->get_array_p(p_, t);
    algebra::axpy(-1.0, p_, ph, pe);
    algebra::zero_boundary(pe);
    double sum_eh_squared = algebra::norm(pe) * std::sqrt(dh.x * dh.y * dh.z);

    std::ofstream pe_file("pe.txt");
    std::ofstream ph_file("ph.txt");
    std::ofstream p__file("p_.txt");
    std::ofstream pbv_file("pbv.txt");
    algebra::print_multi_array<3, double>(pe, pe_file);
    algebra::print_multi_array<3, double>(ph, ph_file);
    algebra::print_multi_array<3, double>(p_, p__file);
    algebra::print_multi_array<3, double>(pbv, pbv_file);

    printf("sum_eh_squared = %.20e\n", sum_eh_squared);
}

} // namespace mykokkos

int main(int argc, char* argv[]) {
    loguru::add_file("Multigrid_P_INFO.log", loguru::Truncate, loguru::Verbosity_INFO);
    loguru::add_file("Multigrid_P_WARNING.log", loguru::Truncate, loguru::Verbosity_WARNING);
    loguru::add_file("Multigrid_P_WATCH.log", loguru::Truncate, loguru::Verbosity_WATCH);
    loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
    Kokkos::initialize(argc, argv);
    std::vector<int> Ns{8, 16, 32, 64, 128, 256, 512};
    std::vector<int> Nt{10};
    for (auto& ns : Ns) {
        std::string msg = fmt::format("Result for ns = {}:", ns);
        std::cout << msg << std::endl;
        for (auto& nt : Nt) {
            std::string msg = fmt::format("           nt = {}:", nt);
            std::cout << msg << std::endl;
            mykokkos::mg3D_p_solver(ns, nt);
        }
    }
    Kokkos::finalize();
    return 0;
}