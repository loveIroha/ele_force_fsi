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
#include <PhysicsSolver/SolidSolver/DiskDriven/DiskDriven.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <catch.hpp>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;
using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::DiskDriven;

double3 function_velocities(const double3& x) {
    double3 f = x;
    return f;
}

double3 function_forces(const double3& x) { return make_double3(1.0, 1.0, 1.0); }

int test_ibm_interpolation_and_distribution() {
    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for immersed body
    int3          dim = {32, 32, 32};
    dolfin::Point p2(0.5, 0.4, 0.3);
    dolfin::Point p3(0.9, 0.8, 0.7);
    auto          solid_mesh        = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(
        dolfin::BoxMesh::create({p2, p3}, {dim.x, dim.y, dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    // Define mesh for background domain
    double  nu           = 0.01;
    double  rho          = 1.0;
    double  dt           = 0.01;
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

    auto& solid_forces     = solid_mesh->create_function("forces");
    auto& fluid_forces     = fluid_mesh_2->create_function("forces");
    auto& solid_velocities = solid_mesh->create_function("velocities");
    auto& fluid_velocities = fluid_mesh_2->create_function("velocities");

    eli->get_fluid_mesh()->list_functions();
    eli->get_solid_mesh()->list_functions();

    // Set velocity on background domain
    fluid_mesh_2->set_function("velocities", function_velocities);

    // Set force on solid domain
    solid_mesh->set_function("forces", function_forces);

    // Interpolation velocity from background domain to solid domain
    eli->interpolate_velocity(solid_velocities, fluid_velocities, solid_mesh->get_quadrature_rules());

    // Distribution force from solid domain to background domain
    eli->distribute_force(fluid_forces, solid_forces, solid_mesh->get_quadrature_rules());

    // Record the results.
    fluid_solver->record(fluid_forces, fluid_velocities, 0.0);
    solid_solver->record(solid_forces, solid_velocities, 0.0);

    return 1;
}

TEST_CASE("test_ibm_interpolation_and_distribution", "[long]") { REQUIRE(test_ibm_interpolation_and_distribution()); }