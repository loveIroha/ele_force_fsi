/**
 * @file time_immersed_boundary_method2.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/NavierStokesFlow/ProjectionSchemeGPU.h>
// #include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <AlgebraSolver/LinearProblem.h>
#include <AlgebraSolver/LinearSolver.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <PhysicsSolver/SolidSolver/DiskDriven/DiskDriven.h>
#include <catch.hpp>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;

using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::DiskDriven;

int time_immersed_boundary_method2() {
    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    double dt = 0.01;
    double nu = 0.04;

    // Define meshes.
    std::cout << "Reading background mesh and immersed mesh.\n";
    auto    domain      = std::make_shared<mshr::Sphere>(dolfin::Point(0.6, 0.5, 0.5), 0.2);
    auto    solid_mesh  = std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain, 20));
    double  rho         = 1.0;
    int3    dim_bg      = {65, 65, 65};
    double3 dh_bg       = {0.0, 0.0, 0.0};
    double3 box_size_bg = {1.0, 1.0, 1.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, dh_bg, box_size_bg, nu, rho);
    std::cout << "Done.\n";

    // Define solvers.
    std::cout << "Creating fluid sover and solid solver.\n";
    auto fluid_solver
        = std::make_shared<UserFluidSolver>(fluid_mesh->dim, fluid_mesh->dh, dt, fluid_mesh->mu, fluid_mesh->rho);
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh->get_dolfin_mesh());
    std::cout << "Done.\n";

    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
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
        ibm_problem->explicit_scheme();
        ibm_problem->get_solid_positions(*xn);
        LOG_F(WARNING, "t : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, xn->_data[0].x, xn->_data[0].y,
              xn->_data[0].z);
    }

    return 1;
}

TEST_CASE("time_immersed_boundary_method2", "[long]") { REQUIRE(time_immersed_boundary_method2()); }