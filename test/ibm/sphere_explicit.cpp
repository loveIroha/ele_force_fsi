/**
 * @file test_ibm.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief time_interpolation_and_distribution
 * @version 0.1
 * @date 2021-12-12
 * @date 2023-05-13
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/LidDrivenFlow/LidDrivenFlow.h>
#include <PhysicsSolver/SolidSolver/DiskDriven/DiskDriven.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>

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

using UserFluidSolver = LidDrivenFlow;
using UserSolidSolver = dolfin::DiskDriven;

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

    // Define mesh for immersed body
    std::ostringstream solid_mesh_path;
    solid_mesh_path << "../geometry/temp/sphere_" << solid_mesh_size << ".xdmf";
    auto solid_mesh        = read_mesh(solid_mesh_path.str());
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    // 输出数据路径
    std::ostringstream result_path;
    result_path << "demo_sphere_explicit_" << fluid_mesh_size << "_" << solid_mesh_size << "_" << time_steps << "/";

    // 日志文件路径
    loguru::add_file((result_path.str() + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file((result_path.str() + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for background domain
    double  nu          = 0.01;
    double  rho         = 1.0;
    int3    dim_bg      = {fluid_mesh_size, fluid_mesh_size, fluid_mesh_size};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {1.0, 1.0, 1.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, origin_bg, box_size_bg, nu, rho);

    // Define solid solver and fluid solver
    auto fluid_solver = std::make_shared<UserFluidSolver>(
        fluid_mesh->dim, fluid_mesh->dh, dt, fluid_mesh->mu, fluid_mesh->rho,
        result_path.str() + "fluid/velocity/data.pvd", result_path.str() + "fluid/pressure/data.pvd",
        result_path.str() + "fluid/force/data.pvd");
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh_dolfin, result_path.str());

    // Define immersed boundary method solver
    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
            solid_mesh, fluid_mesh, solid_solver, fluid_solver);

    // Define immersed boundary method solver
    for (int i = 0; i < time_steps; i++) {
        ibm_problem->explicit_scheme();
        if (i % 100 == 0) { ibm_problem->record(); }
    }

    return 0;
}