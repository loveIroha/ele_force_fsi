//// @date 2025-01-04
/// @file fsi_real_cooke_membrane.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2025 Ma Pengfei
///
/// @brief Cook's membrane
///
///

#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/CookMembrane3D/CookMembrane.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/StaticFlow.h>
#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>
using LocalFluidSolver = stokes_flow::StaticFlow<DIM>;
using LocalSolidSolver = dolfin::CookMembraneSolver;

namespace param {

int    N_s  = 0;
int    N_bg = 64;
int    Nt   = 160000;
double T    = 1.0;
double muf  = 0.04;
double rho  = 1.0;

std::string               path = "./";
std::shared_ptr<JsonFile> json = nullptr;

double nv_stab = 0.5;
double kappa   = 1e300;
double beta    = 1e300;
double G_T     = 1.0;
double G_L     = 1.0;
double E_L     = 1.0;

std::string id        = "";
std::string mesh_path = "";

}; // namespace param

int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary) {
    int                  Nx = param::N_bg;
    int                  Ny = param::N_bg;
    int                  Nz = param::N_bg;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 10.0;
    double                  Ly = 10.0;
    double                  Lz = 10.0; // TODO: it can be 1.0?
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
        auto solid_solver = std::make_shared<LocalSolidSolver>(solid_mesh->get_dolfin_mesh(), nullptr,
                                                               solid_mesh_boundary, param::kappa, param::beta,
                                                               param::G_T, param::G_L, param::E_L, param::path);

        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void cook_membrane() {

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
    std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material = nullptr;

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
    param::beta      = param::json->get_value<double>("beta");
    param::G_T       = param::json->get_value<double>("G_T");
    param::G_L       = param::json->get_value<double>("G_L");
    param::E_L       = param::json->get_value<double>("E_L");
    param::rho       = param::json->get_value<double>("rho");
    param::id        = param::json->get_value<std::string>("id");
    param::nv_stab   = param::json->get_value<double>("nv_stab");
    param::kappa     = 2 * param::G_T * (1 + param::nv_stab) / (3 * (1 - 2 * param::nv_stab));
    param::id        = param::json->get_value<std::string>("id");
    param::mesh_path = param::json->get_value<std::string>("mesh_path");
    param::path      = param::json->get_value<std::string>("path_output") + param::id + "/";

    if (!copy_json_to(config_file, param::path)) { std::cout << "成功复制配置文件！" << std::endl; }

    Kokkos::initialize(argc, argv);
    cook_membrane();
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}
