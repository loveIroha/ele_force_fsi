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
#include <PhysicsSolver/SolidSolver/RealLeftVentricle/RealLeftVentricleSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/StaticFlow.h>
#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::StaticFlow<DIM>;
using LocalSolidSolver = dolfin::RealLeftVentricleSolver;

namespace param {

int    N_s  = 0;
int    N_bg = 64;
int    Nt   = 160000;
double T    = 1.0;
double muf  = 0.04;
double rho  = 1.0;

std::string               path  = "./";
std::shared_ptr<JsonFile> json  = nullptr;
double                    kappa = 1e300;
double                    beta  = 1e300;

std::string id        = "";
std::string mesh_path = "";

std::shared_ptr<dolfin::MeshFunction<double>> f00 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f01 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> f02 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> s00 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> s01 = nullptr;
std::shared_ptr<dolfin::MeshFunction<double>> s02 = nullptr;
}; // namespace param

int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary) {
    int                  Nx = param::N_bg;
    int                  Ny = param::N_bg;
    int                  Nz = param::N_bg;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 13.0;
    double                  Ly = 13.0;
    double                  Lz = 13.0;
    std::array<double, DIM> L{Lx, Ly, Lz};

    double rho = param::rho;
    {
        loguru::add_file((param::path + "INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
        loguru::add_file((param::path + "WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
        loguru::add_file((param::path + "WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_WATCH);
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

        // 创建背景网格
        auto domain_mesh
            = std::make_shared<BackgroundMesh3D<3>>(param::Nt, dim_bg, L, param::T, param::rho, param::muf);

        // 创建流体求解器
        auto fluid_solver = StokesFlowFactory<LocalFluidSolver>::create(dim_bg, param::Nt, L, param::T, param::rho,
                                                                        param::muf, param::path);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver = std::make_shared<LocalSolidSolver>(
            solid_mesh->get_dolfin_mesh(), solid_mesh_material, solid_mesh_boundary, param::f00, param::f01, param::f02,
            param::s00, param::s01, param::s02, param::kappa, param::beta, param::path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void left_ventricle() {

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1(param::mesh_path + "mesh_scale.xdmf");
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }

    // 读取边界文件
    auto solid_mesh_boundary
        = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, param::mesh_path + "boundaries.xml");
    std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material = nullptr;

    // 读取纤维文件
    param::f00 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "fibers_0.xml");
    param::f01 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "fibers_1.xml");
    param::f02 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "fibers_2.xml");
    param::s00 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "sheets_0.xml");
    param::s01 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "sheets_1.xml");
    param::s02 = std::make_shared<dolfin::MeshFunction<double>>(solid_mesh_dolfin, param::mesh_path + "sheets_2.xml");

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_material, solid_mesh_boundary);
}

int main(int argc, char* argv[]) {

    const std::string config_file = argv[1];
    std::cout << "配置文件为：" << config_file << std::endl;

    param::json      = std::make_shared<JsonFile>(config_file);
    param::N_bg      = param::json->get_value<double>("Nb");
    param::Nt        = param::json->get_value<double>("Nt");
    param::T         = param::json->get_value<double>("T");
    param::muf       = param::json->get_value<double>("muf");
    param::kappa     = param::json->get_value<double>("kappa");
    param::beta      = param::json->get_value<double>("beta");
    param::id        = param::json->get_value<std::string>("id");
    param::mesh_path = param::json->get_value<std::string>("mesh_path");
    param::path      = param::json->get_value<std::string>("path_output") + param::id + "/";

    if (!copy_json_to(config_file, param::path)) { std::cout << "成功复制配置文件！" << std::endl; }

    printf("模拟参数为：Ns: %d, Nb: %d, Nt: %d, T: %f, muf: %f, kappa: %f, beta: %f\n", param::N_s, param::N_bg,
           param::Nt, param::T, param::muf, param::kappa, param::beta);
    Kokkos::initialize(argc, argv);
    left_ventricle();
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}