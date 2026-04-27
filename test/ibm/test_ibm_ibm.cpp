/**
 * @file time_immersed_boundary_method.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <AlgebraSolver/LinearProblem.h>
#include <AlgebraSolver/LinearSolver.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/FluidSolver/FluidSolver.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/SolidSolver/DiskDriven/DiskDriven.h>
#include <catch.hpp>

using SolidSolver = dolfin::DiskDriven;

int time_immersed_boundary_method() {
    // loguru::add_file("test_EL_interactor_everything.log", loguru::Append,
    // loguru::Verbosity_MAX); loguru::add_file("test_EL_interactor_warning.log",
    // loguru::Append, loguru::Verbosity_WARNING); loguru::g_stderr_verbosity =
    // loguru::Verbosity_FATAL;

    double dt = 0.01;
    double nu = 0.04;

    std::cout << "Reading background mesh and immersed mesh.\n";
    int3 dim        = {8, 8, 8};
    auto domain     = std::make_shared<mshr::Sphere>(dolfin::Point(0.6, 0.5, 0.5), 0.2);
    auto solid_mesh = std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain, 10));
    auto fluid_mesh = std::make_shared<BackgroundMesh>(dim, dolfin::Point(0, 0, 0), dolfin::Point(1, 1, 1));
    std::cout << "Done.\n";

    std::cout << "Creating fluid sover and solid solver.\n";
    auto fluid_solver = std::make_shared<dolfin::FluidSolver>(fluid_mesh->get_dolfin_mesh(), dt, nu);
    auto solid_solver = std::make_shared<SolidSolver>(solid_mesh->get_dolfin_mesh());
    std::cout << "Done.\n";

    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<SolidSolver, dolfin::FluidSolver, StdVector<double, double3>>>(
            solid_mesh, fluid_mesh, solid_solver, fluid_solver);

    std::cout << "Create Newton solver.\n";
    auto bicgstab = std::make_shared<BiCGSTAB<StdVector<double, double3>>>(ibm_problem->unkown_size() / 3);
    bicgstab->set_tolerance(1e-3);
    NewtonSolver<StdVector<double, double3>> ns(bicgstab);
    ns.max_nonlinear_tolerance = 1e-4;
    ns.max_nonlinear_iteration = 20;
    std::cout << "Done.\n";

    std::cout << "Create variables.\n";
    auto b    = std::make_shared<StdVector<double, double3>>(); // x_{n-1}
    auto xn   = std::make_shared<StdVector<double, double3>>(); // x_{n}
    auto xn_1 = std::make_shared<StdVector<double, double3>>();
    b->resize(ibm_problem->unkown_size() / 3);
    xn->resize(ibm_problem->unkown_size() / 3);
    xn_1->resize(ibm_problem->unkown_size() / 3);
    std::cout << "Done.\n";

    ibm_problem->get_solid_positions(*xn);
    *xn_1 = *xn;

    for (size_t i = 0; i < 1000; i++) {
        IBTimer timer("nonlinear test.");

        // xn = 2*xn - xn_1;
        // xn = xn_1;
        // initial guess for xn;
        xn->axpy(-2.0, *xn, *xn_1); // -2.0*xn + xn_1
        xn->axpy(-2.0, *xn, *xn);   // -2.0*xn + xn
        *xn_1 = *xn;

        auto nonlinear_result = ns.Solve(ibm_problem, xn, b);
        ibm_problem->advance(*xn);
        ibm_problem->record();

        LOG_F(WARNING, "xn: %.12e, %.12e, %.12e", xn->_data[0].x, xn->_data[0].y, xn->_data[0].z);
        LOG_F(WARNING, "Noninear solver successful? %d", nonlinear_result.first);
        LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);
    }
    return 1;
}

TEST_CASE("time_immersed_boundary_method", "[long]") { REQUIRE(time_immersed_boundary_method()); }