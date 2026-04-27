/// @date 2023-09-27
/// @file StokesFlow3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#include <PhysicsSolver/StokesFlow3D/Kokkos/StokesFlow3D_kokkos.h>
#include <nlohmann/json.hpp>

void benchmark_stokes_flow_3D(int Ns, int Nt) {
    int    Nx  = Ns;
    int    Ny  = Ns;
    int    Nz  = Ns;
    double Lx  = 1.0;
    double Ly  = 1.0;
    double Lz  = 1.0;
    double Lt  = 1.0;
    double rho = 1.0;
    double mu  = 1.0;

    const std::array<int, 3>    N = {Nx, Ny, Nz};
    const std::array<double, 3> L = {Lx, Ly, Lz};

    stokes_flow::StokesFlow<3> stokes(Nt, N, L, Lt, rho, mu);
    stokes.main_stokes();
}
int main(int argc, char* argv[]) {
    loguru::add_file("test_EL_interactor_INFO.log", loguru::Truncate, loguru::Verbosity_INFO);
    loguru::add_file("qtest_EL_interactor_WARNING.log", loguru::Truncate, loguru::Verbosity_WARNING);
    loguru::add_file("qtest_EL_interactor_WATCH.log", loguru::Truncate, loguru::Verbosity_WATCH);
    loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

    Kokkos::initialize(argc, argv);

    std::vector<int> Nt = {8};
    std::vector<int> Ns = {8};
    for (const auto& ns : Ns) {
        for (const auto& nt : Nt) {
            benchmark_stokes_flow_3D(ns, nt);
        }
    }
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}