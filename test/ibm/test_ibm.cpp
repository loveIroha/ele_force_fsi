/**
 * @file test_ibm.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief test the immersed boundary method
 * @version 0.1
 * @date 2021-12-16
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <ElerianLagrangianInteraction.h>
#include <FluidSolver/FluidSolver.h>
#include <ImmersedBoundaryMethod.h>
#include <SolidSolver/SolidSolver.h>
#include <catch.hpp>
#include <mshr.h>

// initial velocities
double3 function_expression(const double3& x) { return {0.0, 0.0, 0.0}; }

// initial solid positions
double3 function_positon(const double3& x) { return x; }

int test_ibm() {
    loguru::add_file("test_ibm_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_ibm_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    std::cout << "Create solid mesh.\n";
    auto domain     = std::make_shared<mshr::Sphere>(dolfin::Point(0.6, 0.5, 0.5), 0.2);
    auto solid_mesh = std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain, 10));

    std::cout << "Create background mesh.\n";
    int3 dim        = {32, 32, 32};
    auto fluid_mesh = std::make_shared<BackgroundMesh>(dim);

    auto fluid_mesh_dolfin = fluid_mesh->get_dolfin_mesh();
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    std::cout << "Set physics parameters.\n";
    double mu = 0.1;
    double nu = 0.01;
    double dt = 0.01;

    auto fluid_solver = std::make_shared<dolfin::FluidSolver>(fluid_mesh_dolfin, dt, nu);
    auto solid_solver = std::make_shared<dolfin::SolidSolver>(solid_mesh_dolfin, mu);
    auto eli = std::make_shared<ElerianLagrangianInteraction<StdVector<double, double3>>>(fluid_mesh, solid_mesh);

    auto& solid_forces     = solid_mesh->create_function("forces");
    auto& fluid_forces     = fluid_mesh->create_function("forces");
    auto& solid_velocities = solid_mesh->create_function("velocities");
    auto& fluid_velocities = fluid_mesh->create_function("velocities");
    auto& solid_positions  = solid_mesh->create_function("positions");

    eli->get_fluid_mesh()->list_functions();
    eli->get_solid_mesh()->list_functions();

    fluid_mesh->set_function("velocities", function_expression);
    solid_mesh->set_function("positions", function_positon);

    std::vector<double4> euler_quadrature_rules(solid_mesh->get_quadrature_rules().size());
    solid_mesh->update_Euler_quadrature_rules("positions", euler_quadrature_rules);

    for (size_t i = 0; i < 1000; i++) {
        std::cout << "Solve solid equations.\n";
        solid_forces = solid_solver->solveOneStep(solid_positions);

        std::cout << "Distribute forces from solid to fluid.\n";
        eli->distribute_force(fluid_forces, solid_forces, euler_quadrature_rules);

        std::cout << "Solve NS equations.\n";
        fluid_velocities = fluid_solver->solveOneStep(fluid_forces, fluid_velocities);

        std::cout << "Interpolate velocity from fluid to solid.\n";
        eli->interpolate_velocity(solid_velocities, fluid_velocities, euler_quadrature_rules);

        std::cout << "Update solid postions.\n";
        for (size_t i = 0; i < solid_velocities.size(); i++) {
            solid_positions[i].x += dt * solid_velocities[i].x;
            solid_positions[i].x += dt * solid_velocities[i].x;
            solid_positions[i].x += dt * solid_velocities[i].x;
        }

        std::cout << "Update solid positions on quadrature points.\n";
        solid_mesh->update_Euler_quadrature_rules("positions", euler_quadrature_rules);

        std::cout << "Output the results.\n";
        double t = dt * i;
        fluid_solver->record(fluid_velocities, fluid_forces, t);
        solid_solver->record(solid_forces, solid_positions, t);
    }

    auto ibm_problem = std::make_shared<ImmersedBoundaryMethod<StdVector<double, double3>>>(solid_mesh, fluid_mesh);

    return 1;
}

TEST_CASE("test the immersed boundary method.", "[what]") { REQUIRE(test_ibm()); }
