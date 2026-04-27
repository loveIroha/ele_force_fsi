/// @date 2023-09-28
/// @file benchmark_lid_driven.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief 2024-05-02 修改接口
///
///
///

#include <PhysicsSolver/StokesFlow3D/Kokkos/StokesFlow3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/LidDriven.h>

#include <chrono>
#include <iomanip>
#include <sstream>

std::string generate_time_stamp(const std::string appendix = ".log", const std::string prefix = "./log/") {
    auto               now    = std::chrono::system_clock::now();
    auto               time_t = std::chrono::system_clock::to_time_t(now);
    auto               tm     = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H.%M.%S");
    return prefix + oss.str() + appendix;
}

int calculate_demo_lid_driven(int N, int Nt, std::array<int, 6> bc, double rho, double mu) {

    std::string input_demo_file = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json";
    double      Lx              = 1.0;
    double      Ly              = 1.0;
    double      Lz              = 1.0;
    double      Lt              = 100.0;

    int Nx = N;
    int Ny = N;
    int Nz = N;

    auto stokes_flow_solver = StokesFlowFactory<stokes_flow::LidDriven<3>>::create(Nt, {Nx, Ny, Nz}, {Lx, Ly, Lz}, Lt,
                                                                                   rho, mu, "local/", input_demo_file);

    stokes_flow_solver->change_boundary_types(bc);

    while (stokes_flow_solver->_t < stokes_flow_solver->_Lt / 2) {
        printf("%f  %f\n", stokes_flow_solver->_t, stokes_flow_solver->_dt);
        stokes_flow_solver->_t += stokes_flow_solver->_dt;
        stokes_flow_solver->solve();
        stokes_flow_solver->record();
    }

    stokes_flow_solver->set_dt(0.1);
    while (stokes_flow_solver->_t < stokes_flow_solver->_Lt) {
        printf("%f  %f\n", stokes_flow_solver->_t, stokes_flow_solver->_dt);
        stokes_flow_solver->_t += stokes_flow_solver->_dt;
        stokes_flow_solver->solve();
        stokes_flow_solver->record();
    }
    return 0;
}

int convergence_for_stokes_equations() {

    // 生成64种边界类型的组合
    std::array<int, 6>              type         = {1, 1, 1, 1, 1, 1};
    std::vector<std::array<int, 6>> combinations = {type};

    // 生成所有的 N 和 Nt 的组合
    std::vector<int> all_N  = {96};
    std::vector<int> all_Nt = {10};

    // 密度和粘性系数
    std::vector<double> all_rho = {1.0, 2.0, 3.0};
    std::vector<double> all_mu  = {0.01, 0.02, 0.03};

    // 打印所有参数和边界条件的组合
    for (const auto& mu : all_mu)
        for (const auto& rho : all_rho)
            for (const auto& comb : combinations) {
                for (const auto& N : all_N) {
                    for (const auto& Nt : all_Nt) {
                        calculate_demo_lid_driven(N, Nt, comb, rho, mu);
                        LOG_F(WARNING, "N Nt mu rho : %d %d %f %f ", N, Nt, rho, mu);
                        LOG_F(WARNING, "bc: %d %d %d %d %d %d.", comb[0], comb[1], comb[2], comb[3], comb[4], comb[5]);
                    }
                }
            }
    return 0;
}

int main(int argc, char* argv[]) {
    loguru::add_file(generate_time_stamp("INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
    loguru::add_file(generate_time_stamp("WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
    loguru::add_file(generate_time_stamp("WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_WATCH);
    loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

    Kokkos::initialize(argc, argv);

    convergence_for_stokes_equations();

    Kokkos::finalize();
    return 0;
}