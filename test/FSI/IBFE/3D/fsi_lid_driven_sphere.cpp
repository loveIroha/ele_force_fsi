/// @date 2023-10-22
/// @file fsi_disk_driven_lid.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven3D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow3D/LidDriven.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::LidDriven<DIM>;
using LocalSolidSolver = dolfin::DiskDriven;


int fsi_simulation(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, int Ns, int Nf, int Nt, double T, double mu_f,
                   double mu_s, double lamb) {

    int                  Nx = Nf;
    int                  Ny = Nf;
    int                  Nz = Nf;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 1.0;
    double                  Ly = 1.0;
    double                  Lz = 1.0;
    std::array<double, DIM> L{Lx, Ly, Lz};

    double     rho  = 1.0;
    const auto path = fmt::format("lid_driven_sphere_{}_{}_{}_{:.1f}_with_{:.2e}_{:.2e}_{:.2e}/", Ns, Nf, Nt, T, mu_f, mu_s, lamb);
    {
        ScopeProfiler _{path.c_str()};
        loguru::add_file((path + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
        loguru::add_file((path + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
        loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

        // 创建背景网格
        auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(Nt, dim_bg, L, T, rho, mu_f);

        // 创建流体求解器
        auto fluid_solver = StokesFlowFactory<LocalFluidSolver>::create(dim_bg, Nt, L, T, rho, mu_f, path);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver = std::make_shared<LocalSolidSolver>(solid_mesh->get_dolfin_mesh(), mu_s, lamb, path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void disk_lid_driven(int Ns, int Nf, int Nt, double T, double mu_f, double mu_s, double lamb) {
    // 固体网格路径
    auto mesh_file = geometry_path("temp/sphere_") + std::to_string(Ns);

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1(mesh_file + ".xdmf");
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, Ns, Nf, Nt, T, mu_f, mu_s, lamb);
}

// 命令行参数解析
auto parse_arguments(int argc, char* argv[]) {
    cxxopts::Options options("fsi_disk_driven", "Command line options");
    options.add_options()(
        // 固体步数
        "Ns", "Number of solid discretization.", cxxopts::value<int>()->default_value("30"))(
        // 背景网格步数
        "Nb", "Number of backgrand discretization.", cxxopts::value<int>()->default_value("96"))(
        // 时间步数
        "Nt", "Number of time step.", cxxopts::value<int>()->default_value("1000"))(
        "muf", "流体粘性", cxxopts::value<double>()->default_value("0.01"))(
        "mus", "固体弹性", cxxopts::value<double>()->default_value("0.1"))(
        "lamb", "不可压约束", cxxopts::value<double>()->default_value("100"))(
        // 最终时刻
        "T", "Final time step.", cxxopts::value<double>()->default_value("10"))(

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
    double mus       = arguments["mus"].as<double>();
    double lamb       = arguments["lamb"].as<double>();

    std::cout << "Ns: " << N_s << std::endl;
    std::cout << "Nb: " << N_bg << std::endl;
    std::cout << "Nt: " << Nt << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "muf: " << muf << std::endl;
    std::cout << "mus: " << mus << std::endl;
    std::cout << "lamb: " << lamb << std::endl;

    Kokkos::initialize(argc, argv);
    disk_lid_driven(N_s, N_bg, Nt, T, muf, mus, lamb);
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}


// taskset -c 35 nohup ./test/fsi_lid_driven_sphere --Nt 20000 -T 10 --Nb 64 --Ns 30 --mus 0.1 --muf 0.01 --lamb 100 &
// disown
// taskset -c 36 nohup ./test/fsi_lid_driven_sphere --Nt 10000 -T 10 --Nb 64 --Ns 30 --mus 0.1 --muf 0.01 --lamb 100 &
// disown
// taskset -c 37 nohup ./test/fsi_lid_driven_sphere --Nt 5000 -T 10 --Nb 64 --Ns 30 --mus 0.1 --muf 0.01 --lamb 100 &
// disown
