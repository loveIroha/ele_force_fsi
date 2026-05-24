/// @date 2024-04-20
/// @file box_tube_flow.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief 
/// 
///

#include <PhysicsSolver/StokesFlow3D/TubeFlow.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>

void fun(int Ns, int Nt, double rho, double mu_f) {
    double T = 0.6;
    int Nx = Ns;
    int Ny = Ns;
    int Nz = Ns*1.5;

    double                  Lx = 10.0;
    double                  Ly = 10.0;
    double                  Lz = 16.0;
    std::array<double, DIM> L{Lx, Ly, Lz};
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};
    std::string input_demo_file = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json"; // FIXME: 删除这一行
    auto fluid_solver = StokesFlowFactory<stokes_flow::TubeFlow<3>>::create(Nt, dim_bg, L, T,
                                                                                    rho, mu_f, "cfd/", input_demo_file);

    while (fluid_solver->_t < T)
    {
        if (fluid_solver->_t > 0.185) { fluid_solver->set_dt(1.0e-3);} // 可以在程序运行过程中设置时间步长
        printf("%f  %f\n", fluid_solver->_t, fluid_solver->_dt);
        fluid_solver->_t += fluid_solver->_dt;
        fluid_solver->solve();
        fluid_solver->record();
    }
}

int main(int argc, char* argv[]) {
    loguru::add_file("Tube_Flow_INFO.log", loguru::Truncate, loguru::Verbosity_INFO);
    loguru::add_file("Tube_Flow_WARNING.log", loguru::Truncate, loguru::Verbosity_WARNING);
    loguru::add_file("Tube_Flow_WATCH.log", loguru::Truncate, loguru::Verbosity_1);
    loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
    Kokkos::initialize(argc, argv);

    fun(96, 100, 1.0, 0.04);
    
    Kokkos::finalize();
    return 0;
}
