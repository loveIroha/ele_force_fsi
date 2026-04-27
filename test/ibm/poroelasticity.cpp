/**
 * @file test_ibm.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief time_interpolation_and_distribution
 * @version 0.1
 * @date 2021-12-12
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/FluidPoroelasticInteractionSolver.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/LidDrivenFlow/LidDrivenFlow.h>
#include <PhysicsSolver/LidDrivenFlow/StaticFlow.h>
#include <PhysicsSolver/SolidSolver/BarProblem/BarSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <catch.hpp>
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
using UserSolidSolver = dolfin::BarSolver;

std::shared_ptr<ImmersedMesh> read_mesh(std::string filename) {
    dolfin::XDMFFile mesh_file_1(filename.c_str());
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto solid_mesh = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);
    return solid_mesh;
}

int test_ibm_interpolation_and_distribution() {
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for immersed body
    auto solid_mesh        = read_mesh("../geometry/temp/bar_20.xdmf");
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    // Define mesh for background domain
    double  nu          = 0.04;
    double  rho         = 1.0;
    double  dt          = 5e-6;
    int3    dim_bg      = {129, 129, 129};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {10.0, 10.0, 10.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, origin_bg, box_size_bg, nu, rho);

    // Define solid solver and fluid solver
    auto fluid_solver
        = std::make_shared<UserFluidSolver>(fluid_mesh->dim, fluid_mesh->dh, dt, fluid_mesh->mu, fluid_mesh->rho);
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh->get_dolfin_mesh());

    // Define interplator and distributor
    auto eli = std::make_shared<ElerianLagrangianInteraction<StdVector<double, double3>>>(fluid_mesh, solid_mesh);

    // Define immersed boundary method solver
    auto ibm_problem = std::make_shared<
        FluidPoroelasticInteractionSolver<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
        solid_mesh, fluid_mesh, solid_solver, fluid_solver);

    // Initialize variables
    auto xn = std::make_shared<StdVector<double, double3>>();
    xn->resize(ibm_problem->unkown_size() / 3);

    // Define immersed boundary method solver
    for (size_t i = 0; i < 200000; i++) {
        ibm_problem->explicit_scheme();
        if (i % 100 == 0) { ibm_problem->record(); }
        ibm_problem->get_solid_positions(*xn);
        LOG_F(WARNING, "t : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, xn->_data[0].x, xn->_data[0].y,
              xn->_data[0].z);
    }

    return 1;
}

TEST_CASE("test_ibm_interpolation_and_distribution", "[long]") { REQUIRE(test_ibm_interpolation_and_distribution()); }