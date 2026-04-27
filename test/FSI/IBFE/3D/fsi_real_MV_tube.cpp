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
#include <PhysicsSolver/SolidSolver/RealMitralValve/RealMitralValveSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/TubeFlow.h>

#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::TubeFlow<DIM>;
using LocalSolidSolver = dolfin::RealMitralValveSolver;

namespace param {

int    N_s    = 0;
int    N_bg  = 64;
int    Nt    = 160000;
double T     = 1.0;
double muf   = 0.04;
double rho   = 1.0;

std::string               path            = "./";
std::shared_ptr<JsonFile> json            = nullptr;
double                    kappa           = 1e300;
double                    beta            = 1e300;
double                    t_start_closing = 0.185;
double                    dilation_radius = 0.1;

double posterior_papillary_x = 0.0;
double posterior_papillary_y = 0.0;
double posterior_papillary_z = 0.0;
double anterior_papillary_x  = 0.0;
double anterior_papillary_y  = 0.0;
double anterior_papillary_z  = 0.0;
std::string id = "";
std::string mesh_path = "";

// 
std::shared_ptr<dolfin::MeshFunction<double>> f00 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f01 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f02 = nullptr;
}; // namespace param

int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary) {
    int                  Nx = param::N_bg;
    int                  Ny = param::N_bg;
    int                  Nz = 1.5 * param::N_bg;
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
        auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(param::Nt, dim_bg, L, param::T, param::rho, param::muf);

        // 创建流体求解器
        auto fluid_solver = StokesFlowFactory<LocalFluidSolver>::create(
            dim_bg, param::Nt, L, param::T, param::rho, param::muf, param::path);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver = std::make_shared<LocalSolidSolver>(
            solid_mesh->get_dolfin_mesh(), solid_mesh_material, solid_mesh_boundary, 
            param::f00, param::f01, param::f02,
            param::kappa, 
            param::beta, 
            param::t_start_closing, 
            param::dilation_radius, 
            param::posterior_papillary_x,
            param::posterior_papillary_y, 
            param::posterior_papillary_z, 
            param::anterior_papillary_x,
            param::anterior_papillary_y, 
            param::anterior_papillary_z, param::path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void mitral_valve() {

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1(param::mesh_path + param::json->get_value<std::string>("file_mesh"));
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }

    // 读取边界文件
    auto solid_mesh_boundary = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, param::mesh_path + param::json->get_value<std::string>("file_boundaries"));
    printf("solid_mesh_dolfin: %d\n", solid_mesh_dolfin->topology().dim());
    auto solid_mesh_material = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, param::mesh_path + param::json->get_value<std::string>("file_materials"));

    // 读取纤维文件
    param::f00 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, param::mesh_path + param::json->get_value<std::string>("file_fiber_0"));
    param::f01 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, param::mesh_path + param::json->get_value<std::string>("file_fiber_1"));
    param::f02 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, param::mesh_path + param::json->get_value<std::string>("file_fiber_2"));

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_material, solid_mesh_boundary);
}

int main(int argc, char* argv[]) {

    const std::string config_file = argv[1];
    std::cout << "配置文件为：" << config_file << std::endl;

    param::json  = std::make_shared<JsonFile>(config_file);
    param::N_bg  = param::json->get_value<double>("Nb");
    param::Nt    = param::json->get_value<double>("Nt");
    param::T     = param::json->get_value<double>("T");
    param::muf   = param::json->get_value<double>("muf");
    param::kappa = param::json->get_value<double>("kappa");
    param::beta  = param::json->get_value<double>("beta");
    param::id    = param::json->get_value<std::string>("id");
    param::mesh_path = param::json->get_value<std::string>("mesh_path");
    param::path = param::json->get_value<std::string>("path_output")  + param::id + "/";

    param::t_start_closing = param::json->get_value<double>("t_start_closing");
    param::dilation_radius = param::json->get_value<double>("dilation_radius");

    param::posterior_papillary_x = param::json->get_value<double>("posterior_papillary_x");
    param::posterior_papillary_y = param::json->get_value<double>("posterior_papillary_y");
    param::posterior_papillary_z = param::json->get_value<double>("posterior_papillary_z");

    param::anterior_papillary_x = param::json->get_value<double>("anterior_papillary_x");
    param::anterior_papillary_y = param::json->get_value<double>("anterior_papillary_y");
    param::anterior_papillary_z = param::json->get_value<double>("anterior_papillary_z");


    if (!copy_json_to(config_file, param::path)) { std::cout << "成功复制配置文件！" << std::endl; }

    printf("模拟参数为：Ns: %d, Nb: %d, Nt: %d, T: %f, muf: %f, kappa: %f, beta: %f\n", param::N_s, param::N_bg, param::Nt, param::T, param::muf,
           param::kappa, param::beta);
    Kokkos::initialize(argc, argv);
    mitral_valve();
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}