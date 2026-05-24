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
#include <PhysicsSolver/SolidSolver/RealLeftVentricleMitralValve/RealLeftVentricleMitralValveSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/TubeFlowStatic.h>

#include <config.h>
#include <fmt/core.h>
#include <io/parameters.h>

using LocalFluidSolver = stokes_flow::TubeFlow<DIM>;
using LocalSolidSolver = dolfin::RealMitralValveSolver;

namespace param {

std::string               path = "./";    // 输出路径
std::shared_ptr<JsonFile> json = nullptr; // 配置文件

double kappa = 1e300;   // 不可压
double beta  = 1e300;   // 固定位移
int    Nf    = 80;
int    Nt    = 10000;
double T     = 2.0;
double muf   = 0.4;
double rho   = 1.0;

double t_start         = 0.0;  // 瓣膜打开阶段
double t_end_loading   = 0.4;  // 内壁压强达到8mmHg
double t_end_diastole  = 0.8;  // 舒张末期，开始增加钙离子浓度
double t_end_isovolume = 0.84; // 等容收缩末期，心室压强达85mmHg，0.95s
double t_end_systole   = 1.2;  // 收缩末期，钙离子浓度降为零
double t_end           = 2.0;  // 模拟结束
double t_period        = 2.0;
int    num_cycles      = 1;
double p_load          = 8.0;  // mmHg

std::shared_ptr<dolfin::FiberDirections> f0 = nullptr;
std::shared_ptr<dolfin::FiberDirections> s0 = nullptr;
}; // namespace param

int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_material,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary) {
    std::array<int, DIM>    dim_bg{param::Nf, param::Nf, param::Nf};
    std::array<double, DIM> L{17.0, 16.0, 16.5};
    {
        // 日志文件
        loguru::add_file((param::path + "INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
        loguru::add_file((param::path + "WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
        loguru::add_file((param::path + "WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_1);
        loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

        // 创建背景网格
        auto domain_mesh
            = std::make_shared<BackgroundMesh3D<3>>(param::Nt, dim_bg, L, param::T, param::rho, param::muf);

        // 创建流体求解器
        std::string input_demo_file
            = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json"; // FIXME: 删除这一行
        auto fluid_solver = StokesFlowFactory<LocalFluidSolver>::create(param::Nt, dim_bg, L, param::T, param::rho,
                                                                        param::muf, param::path, input_demo_file);

        // 创建固体网格
        auto solid_mesh = std::make_shared<ImmersedMeshP1>(solid_mesh_dolfin);

        // 创建固体求解器
        auto solid_solver = std::make_shared<LocalSolidSolver>(solid_mesh->get_dolfin_mesh(), solid_mesh_material,
                                                               solid_mesh_boundary, param::f0, param::s0, param::kappa,
                                                               param::beta, param::t_end_diastole, param::path);

        // 运行算例
        fsi_lid_simulation_explicit(domain_mesh, fluid_solver, solid_mesh, solid_solver);
    }

    return 0;
}

void fsi_simulation() {
    // 固体网格路径
    auto mesh_file = geometry_path("LVMV/mesh/");

    // 读取网格文件
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    dolfin::XDMFFile mesh_file_1(mesh_file + param::json->get_value<std::string>("file_mesh"));
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    LOG_F(INFO, "Load solid mesh.");

    // 读取边界文件
    auto solid_mesh_boundary = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_boundaries"));
    auto solid_mesh_material = std::make_shared<dolfin::MeshFunction<std::size_t>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_materials"));
    LOG_F(INFO, "Load boundary and material marker of solid mesh.");

    // 读取纤维文件
    auto f00 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_0"));
    auto f01 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_1"));
    auto f02 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_fiber_2"));
    auto s00 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_sheet_0"));
    auto s01 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_sheet_1"));
    auto s02 = std::make_shared<dolfin::MeshFunction<double>>(
        solid_mesh_dolfin, mesh_file + param::json->get_value<std::string>("file_sheet_2"));
    param::f0 = std::make_shared<dolfin::FiberDirections>(f00, f01, f02);
    param::s0 = std::make_shared<dolfin::FiberDirections>(s00, s01, s02);
    LOG_F(INFO, "Load fibers and sheets.");

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_material, solid_mesh_boundary);
}

int main(int argc, char* argv[]) {

    const std::string config_file = argv[1];
    LOG_F(INFO, "配置文件为：%s", config_file.c_str());

    param::json  = std::make_shared<JsonFile>(config_file);
    param::Nf    = param::json->get_value<double>("Nb");
    param::Nt    = param::json->get_value<double>("Nt");
    param::T     = param::json->get_value<double>("T");
    param::muf   = param::json->get_value<double>("muf");
    param::kappa = param::json->get_value<double>("kappa");
    param::beta  = param::json->get_value<double>("beta");

    param::t_end_diastole = param::json->get_value<double>("t_end_diastole");

    param::path
        = param::json->get_value<std::string>("path_output")
          + fmt::format("discretization-{}_{}_{:.1f}/physical-{:.2e}_{:.2e}_{:.2e}/", 
                        param::Nf, param::Nt, param::T, param::muf, param::kappa, param::beta);

    if (!copy_json_to(config_file, param::path)) { LOG_F(INFO, "成功复制配置文件！"); }

    printf("模拟参数为 Nb: %d, Nt: %d, T: %f, muf: %f, kappa: %f, beta: %f\n",  param::Nf, param::Nt,
           param::T, param::muf, param::kappa, param::beta);

    Kokkos::initialize(argc, argv);
    fsi_simulation();
    Kokkos::finalize();
    printScopeProfiler();
    return 0;
}