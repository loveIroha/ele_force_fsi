/// @date 2023-11-01
/// @file fsi_real_LV_diastole.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///         2023-11-02 真实左心室模型
///         2024-03-12 适应新系统

#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/RealBiVentricleSystole/RealBiVentricleSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/StaticFlow.h>
#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::StaticFlow<DIM>;
using LocalSolidSolver = dolfin::RealBiVentricleSolver;
namespace param {
double diastole_pressure = 10665.789;
double systole_pressure  = 134600.0;
double max_tension       = 842600.0;
double kappa             = 1000000.0;
double beta              = 10000000.0;
}; // namespace param



int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary,
                   std::shared_ptr<dolfin::MeshFunction<double>> f00, std::shared_ptr<dolfin::MeshFunction<double>> f01,
                   std::shared_ptr<dolfin::MeshFunction<double>> f02, std::shared_ptr<dolfin::MeshFunction<double>> s00,
                   std::shared_ptr<dolfin::MeshFunction<double>> s01, std::shared_ptr<dolfin::MeshFunction<double>> s02,
                   int Ns, int Nf, int Nt, double T, double mu_f) {
    int                  Nx = Nf;
    int                  Ny = Nf;
    int                  Nz = Nf;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 14.0;
    double                  Ly = 14.0;
    double                  Lz = 14.0;
    std::array<double, DIM> L{Lx, Ly, Lz};

    double     rho  = 1.0;
    const auto path = fmt::format("real_BV_systole_3D_explicit_{}_{}_{}_{:.1f}_with_{:.2e}_{:.2e}_{:.2e}/", Ns, Nf, Nt,
                                  T, mu_f, param::kappa, param::beta);
    {
        loguru::add_file((path + "INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
        loguru::add_file((path + "WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
        loguru::add_file((path + "WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_WATCH);
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

        // 创建背景网格
        auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(Nt, dim_bg, L, T, rho, mu_f);

        // 创建流体求解器
        auto fluid_solver = StokesFlowFactory<LocalFluidSolver>::create(dim_bg, Nt, L, T, rho, mu_f, path);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver
            = std::make_shared<LocalSolidSolver>(solid_mesh->get_dolfin_mesh(), solid_mesh_boundary, f00, f01, f02, s00,
                                                 s01, s02, param::kappa, param::beta, path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void bar_bending(int Ns, int Nf, int Nt, double T, double mu_f) {
    // 固体网格路径
    auto mesh_file = geometry_path("ventricle/BV/");

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1(mesh_file + "mesh/mesh_scale.xdmf");
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }

    // 读取边界文件
    auto solid_mesh_boundary
        = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, mesh_file + "mesh/boundaries.xml");

    // 读取纤维文件
    auto f00 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/fibers_0.xml");
    auto f01 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/fibers_1.xml");
    auto f02 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/fibers_2.xml");
    auto s00 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/sheets_0.xml");
    auto s01 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/sheets_1.xml");
    auto s02 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, mesh_file + "fibers/sheets_2.xml");

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_boundary, f00, f01, f02, s00, s01, s02, Ns, Nf, Nt, T, mu_f);
}

// 命令行参数解析
auto parse_arguments(int argc, char* argv[]) {
    cxxopts::Options options("fsi_disk_driven", "Command line options");
    options.add_options()(
        // 固体步数
        "Ns", "Number of solid discretization.", cxxopts::value<int>()->default_value("40"))(
        // 背景网格步数
        "Nb", "Number of backgrand discretization.", cxxopts::value<int>()->default_value("64"))(
        // 时间步数
        "Nt", "Number of time step.", cxxopts::value<int>()->default_value("10000"))(
        "muf", "流体粘性", cxxopts::value<double>()->default_value("0.04"))(
        // "mus", "固体弹性", cxxopts::value<double>()->default_value("0.1"))(
        "kappa", "不可压约束", cxxopts::value<double>()->default_value("100000"))(
        "beta", "固定", cxxopts::value<double>()->default_value("10000000"))(
        // 最终时刻
        "T", "Final time step.", cxxopts::value<double>()->default_value("40"))(

        "h,help", "Show help");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return result;
}

int main(int argc, char* argv[]) {

    auto   arguments = parse_arguments(argc, argv);
    int    N_s       = arguments["Ns"].as<int>();
    int    N_bg      = arguments["Nb"].as<int>();
    int    Nt        = arguments["Nt"].as<int>();
    double T         = arguments["T"].as<double>();
    double muf       = arguments["muf"].as<double>();
    param::kappa     = arguments["kappa"].as<double>();
    param::beta      = arguments["beta"].as<double>();
    Kokkos::initialize(argc, argv);
    bar_bending(N_s, N_bg, Nt, T, muf);
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}