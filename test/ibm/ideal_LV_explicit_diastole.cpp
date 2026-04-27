/// @date 2023-05-08
/// @file ideal_left_ventricle_diastole_explicit.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 理想左心室舒张
///
///

#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/LidDrivenFlow/StaticFlow.h>
#include <PhysicsSolver/SolidSolver/IdealLeftVentricle/IdealLeftVentricleSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <config.h>

using namespace pangu;

// 定义数据类型
using PressureType         = StdVector<double, double>;
using VelocityType         = StdVector<double, double3>;
using PressureBoundaryType = StdVector<char, char>;
using VelocityBoundaryType = StdVector<char, char>;

// CUDA数据类型
using GPUPressureType         = GpuVector<double, double>;
using GPUVelocityType         = GpuVector<double, double3>;
using GPUPressureBoundaryType = GpuVector<char, char>;
using GPUVelocityBoundaryType = GpuVector<char, char>;

using UserFluidSolver = StaticFlow;
using UserSolidSolver = dolfin::IdealLeftVentricleSolver;



std::shared_ptr<ImmersedMesh> read_mesh(std::string filename) {
    dolfin::XDMFFile mesh_file_1(filename.c_str());
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto solid_mesh = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);
    return solid_mesh;
}

int main(int argc, char* argv[]) {
    CHECK_F(argc == 5, "必须输入四个参数");

    // 固体网格大小、流体网格大小、时间比率、GPU设备

    // 离散参数
    double T               = 10.0;
    int    solid_mesh_size = std::stoi(argv[1]);
    int    fluid_mesh_size = std::stoi(argv[2]);
    int    time_step_ratio = std::stoi(argv[3]);
    int    time_steps      = T * fluid_mesh_size * time_step_ratio;
    double dt              = T / time_steps;

    // 设置GPU设备
    gpu::find_devices();
    gpu::set_device(std::stoi(argv[4]));

    // 读取固体网格
    auto solid_mesh_path = geometry_path("temp/ventricle/LV/LV_ideal/LV_ideal_");
    solid_mesh_path += std::to_string(solid_mesh_size);
    solid_mesh_path += ".xdmf";
    auto solid_mesh        = read_mesh(solid_mesh_path);
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    auto solid_mesh_boundary_path = geometry_path("temp/ventricle/LV/LV_ideal/LV_ideal_");
    solid_mesh_boundary_path += std::to_string(solid_mesh_size);
    solid_mesh_boundary_path += "_boundary.xml";
    auto solid_mesh_boundaries
        = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, solid_mesh_boundary_path);

    // 输出数据路径
    std::ostringstream result_path;
    result_path << "demo_ilv_explicit_" << fluid_mesh_size << "_" << solid_mesh_size << "_" << time_steps << "/";

    // 日志文件路径
    loguru::add_file((result_path.str() + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file((result_path.str() + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for background domain
    double  nu          = 0.04;
    double  rho         = 1.0;
    int3    dim_bg      = {fluid_mesh_size, fluid_mesh_size, fluid_mesh_size};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {5.0, 5.0, 5.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, origin_bg, box_size_bg, nu, rho);

    // Define solid solver and fluid solver
    auto fluid_solver = std::make_shared<UserFluidSolver>(
        fluid_mesh->dim, fluid_mesh->dh, dt, fluid_mesh->mu, fluid_mesh->rho,
        result_path.str() + "fluid/velocity/data.pvd", result_path.str() + "fluid/pressure/data.pvd",
        result_path.str() + "fluid/force/data.pvd");
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh_dolfin, solid_mesh_boundaries, result_path.str());

    // Define immersed boundary method solver
    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
            solid_mesh, fluid_mesh, solid_solver, fluid_solver);

    // Initialize variables
    auto xn = std::make_shared<StdVector<double, double3>>();
    xn->resize(ibm_problem->unkown_size() / 3);

    // Define immersed boundary method solver
    for (size_t i = 0; i < time_steps; i++) {
        ibm_problem->explicit_scheme();
        if (i % 100 == 0) { ibm_problem->record(); }
        ibm_problem->get_solid_positions(*xn);
        LOG_F(WARNING, "t : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, xn->_data[0].x, xn->_data[0].y,
              xn->_data[0].z);
    }

    return 0;
}