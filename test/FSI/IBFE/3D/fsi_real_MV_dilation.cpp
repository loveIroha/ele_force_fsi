/// @date 2024-04-23
/// @file fsi_real_MV_tube.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/RealMitralValveDilation/RealMitralValveSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/TubeFlowStatic.h>

#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::TubeFlow<DIM>;
using LocalSolidSolver = dolfin::RealMitralValveSolver;

namespace param {
std::string                                   path  = "./";
std::shared_ptr<JsonFile>                     json  = nullptr;
double                                        kappa = 1e300;
double                                        beta  = 1e300;
double                             t_start_closing  = 0.185;
double                             dilation_radius  = 0.1;
std::shared_ptr<dolfin::MeshFunction<double>> f00   = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f01   = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f02   = nullptr;
}; // namespace param



int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary, int Ns, int Nf, int Nt,
                   double T, double mu_f) {
    int                  Nx = Nf;
    int                  Ny = Nf;
    int                  Nz = 1.5 * Nf;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 10.0;
    double                  Ly = 10.0;
    double                  Lz = 16.0;
    std::array<double, DIM> L{Lx, Ly, Lz};

    double rho = 1.0;

    {
        loguru::add_file((param::path + "INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
        loguru::add_file((param::path + "WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
        loguru::add_file((param::path + "WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_WATCH);
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

        // 创建背景网格
        auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(Nt, dim_bg, L, T, rho, mu_f);

        // 创建流体求解器
        std::string input_demo_file = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json"; // FIXME: 删除这一行
        auto fluid_solver = StokesFlowFactory<stokes_flow::TubeFlow<3>>::create(Nt, dim_bg, L, T,
                                                                                    rho, mu_f, param::path, input_demo_file);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver = std::make_shared<LocalSolidSolver>(solid_mesh->get_dolfin_mesh(), solid_mesh_material,
                                                               solid_mesh_boundary, param::f00, param::f01, param::f02,
                                                               param::kappa, param::beta, param::t_start_closing, param::dilation_radius,
                                                               param::path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void mitral_valve(int Ns, int Nf, int Nt, double T, double mu_f) {
    // 固体网格路径
    auto mesh_file = geometry_path("valve/MV/mesh/");

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1(mesh_file + param::json->get_value<std::string>("file_mesh"));
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }
    printf("solid_mesh_dolfin: %d\n", solid_mesh_dolfin->topology().dim());

    // 读取边界文件
    auto solid_mesh_boundary = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_boundaries"));
    printf("solid_mesh_dolfin: %d\n", solid_mesh_dolfin->topology().dim());
    auto solid_mesh_material = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_materials"));

    // 读取纤维文件
    param::f00 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_0"));
    param::f01 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_1"));
    param::f02 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_2"));

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_material, solid_mesh_boundary, Ns, Nf, Nt, T, mu_f);
}

int copy_json_to(const std::string& copy_from, const std::string& copy_to) {
    // 检查目标文件夹是否存在，如果不存在则创建
    if (!std::filesystem::exists(copy_to)) std::filesystem::create_directories(copy_to);

    // 构建目标文件的路径
    std::string destinationFile = copy_to + "/" + copy_from;

    // 打开源文件和目标文件
    std::ifstream sourceStream(copy_from, std::ios::binary);
    std::ofstream destinationStream(destinationFile, std::ios::binary);

    // 检查文件是否成功打开
    if (!sourceStream.is_open() || !destinationStream.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }

    // 从源文件复制到目标文件
    destinationStream << sourceStream.rdbuf();

    // 关闭文件流
    sourceStream.close();
    destinationStream.close();

    return 0;
}

int main(int argc, char* argv[]) {

    const std::string config_file = argv[1];
    std::cout << "配置文件为：" << config_file << std::endl;

    param::json  = std::make_shared<JsonFile>(config_file);
    int    N_s   = param::json->get_value<double>("Ns");
    int    N_bg  = param::json->get_value<double>("Nb");
    int    Nt    = param::json->get_value<double>("Nt");
    double T     = param::json->get_value<double>("T");
    double muf   = param::json->get_value<double>("muf");
    param::kappa = param::json->get_value<double>("kappa");
    param::beta  = param::json->get_value<double>("beta");
    param::t_start_closing = param::json->get_value<double>("t_start_closing");
    param::dilation_radius = param::json->get_value<double>("dilation_radius");

    param::path = param::json->get_value<std::string>("path_output")
                  + fmt::format("physical{}_{}_{}_{:.1f}/discretization{:.2e}_{:.2e}_{:.2e}/dilation_{:.2e}_{:.2e}/", N_s, N_bg, Nt, T, muf,
                                param::kappa, param::beta, param::t_start_closing, param::dilation_radius);

    if (!copy_json_to(config_file, param::path)) { std::cout << "成功复制配置文件！" << std::endl; }

    printf("模拟参数为：Ns: %d, Nb: %d, Nt: %d, T: %f, muf: %f, kappa: %f, beta: %f\n", N_s, N_bg, Nt, T, muf,
           param::kappa, param::beta);
    Kokkos::initialize(argc, argv);
    mitral_valve(N_s, N_bg, Nt, T, muf);
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}