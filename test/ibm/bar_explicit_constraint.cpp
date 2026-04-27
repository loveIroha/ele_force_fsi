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
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/SolidSolver/BarProblem/BarSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <catch.hpp>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;
using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::BarSolver;

int test_ibm_interpolation_and_distribution() {
    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for immersed body
    // double position_bottom = 0.45;
    // double position_left = 0.1;
    int3          dim = {4, 4, 32};
    dolfin::Point p2(0.45, 0.45, 0.1);
    dolfin::Point p3(0.55, 0.55, 0.9);
    auto          solid_mesh        = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(
        dolfin::BoxMesh::create({p2, p3}, {dim.x, dim.y, dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    // Define mesh for background domain
    double  nu           = 0.01;
    double  rho          = 1.0;
    double  dt           = 0.0001;
    int3    dim_bg       = {65, 65, 65};
    double3 dh_bg        = {0.0, 0.0, 0.0};
    double3 box_size_bg  = {1.0, 1.0, 1.0};
    auto    fluid_mesh_2 = std::make_shared<BackgroundMesh2>(dim_bg, dh_bg, box_size_bg, nu, rho);

    // Define solid solver and fluid solver
    auto fluid_solver = std::make_shared<UserFluidSolver>(fluid_mesh_2->dim, fluid_mesh_2->dh, dt, fluid_mesh_2->mu,
                                                          fluid_mesh_2->rho);
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh->get_dolfin_mesh());

    // Define interplator and distributor
    auto eli = std::make_shared<ElerianLagrangianInteraction<StdVector<double, double3>>>(fluid_mesh_2, solid_mesh);

    // Define immersed boundary method solver
    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
            solid_mesh, fluid_mesh_2, solid_solver, fluid_solver);

    // Initialize variables
    auto xn = std::make_shared<StdVector<double, double3>>();
    xn->resize(ibm_problem->unkown_size() / 3);

    // Define immersed boundary method solver
    for (size_t i = 0; i < 20000; i++) {
        ibm_problem->explicit_scheme();
        ibm_problem->constraint_positions();
        if (i % 100 == 0) { ibm_problem->record(); }
        ibm_problem->get_solid_positions(*xn);
        LOG_F(WARNING, "t : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, xn->_data[0].x, xn->_data[0].y,
              xn->_data[0].z);
    }

    return 1;
}

TEST_CASE("test_ibm_interpolation_and_distribution", "[long]") { REQUIRE(test_ibm_interpolation_and_distribution()); }