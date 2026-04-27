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

#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/LinearProblem.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/NavierStokesFlow/ProjectionSchemeGPU.h>
#include <PhysicsSolver/SolidSolver/RealLeftVentricle/RealLeftVentricleSolver.h>
#include <catch.hpp>
#include <loguru/smtp.h>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;

using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::RealLeftVentricleSolver;

int time_immersed_boundary_method2() {
    LOG_SCOPE_FUNCTION(INFO);

    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_info.log", loguru::Append, loguru::Verbosity_INFO);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::add_file("test_EL_interactor_0.log", loguru::Append, loguru::Verbosity_0);
    loguru::add_file("test_EL_interactor_1.log", loguru::Append, loguru::Verbosity_1);
    loguru::add_file("test_EL_interactor_2.log", loguru::Append, loguru::Verbosity_2);
    loguru::add_file("test_EL_interactor_3.log", loguru::Append, loguru::Verbosity_3);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    double dt = 1e-4;
    double nu = 0.04;

    // Define meshes.
    std::cout << "Reading background mesh and immersed mesh.\n";
    double  rho         = 1.0;
    int3    dim_bg      = {129, 129, 129};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {15.0, 15.0, 15.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, origin_bg, box_size_bg, nu, rho);
    std::cout << "Done.\n";

    std::cout << "Reading solid mesh.\n";
    dolfin::XDMFFile mesh_file_1("/mnt/large2/gjh/realistic_left_ventricle/mesh_scale.xdmf");
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto solid_mesh = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);
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
    bicgstab->set_max_iteration(20);
    NewtonSolver<StdVector<double, double3>> ns(bicgstab);
    ns.max_nonlinear_tolerance = 1e-5;
    ns.max_nonlinear_iteration = 100;
    std::cout << "Done.\n";

    std::cout << "Create variables.\n";
    auto b      = std::make_shared<StdVector<double, double3>>(); // x_{n-1}
    auto xn     = std::make_shared<StdVector<double, double3>>(); // x_{n}
    auto xn_1   = std::make_shared<StdVector<double, double3>>();
    auto xn_new = std::make_shared<StdVector<double, double3>>();
    b->resize(ibm_problem->unkown_size() / 3);
    xn->resize(ibm_problem->unkown_size() / 3);
    xn_1->resize(ibm_problem->unkown_size() / 3);
    xn_new->resize(ibm_problem->unkown_size() / 3);
    std::cout << "Done.\n";

    ibm_problem->get_solid_positions(*xn);
    *xn_1   = *xn;
    *xn_new = *xn;

    double dt_ratio         = 1.0;
    auto   nonlinear_result = std::make_pair(true, std::make_pair(0.0, 0));
    while (ibm_problem->_t < 10) {
        // LOG_SCOPE_F(INFO, "nonlinear test.");
        // LOG_F(WARNING, "t : %.12e, dt : %.12e", ibm_problem->_t, ibm_problem->_dt);
        // smtp::email_notification("mapengfei@mail.nwpu.edu.cn",
        //                          std::string(" first order element with lumping mass matrix. \n\n")
        //                              + std::string("special constraint on base.")
        //                              + "current time : " + std::to_string(ibm_problem->_t) + std::string("\n")
        //                              + "delta t : " + std::to_string(ibm_problem->_dt) + std::string("\n"));

        xn_new->axpy(-(1.0 + dt_ratio) / dt_ratio, *xn, *xn_1);
        xn_new->axpy(-1.0 - dt_ratio, *xn_new, *xn_new);

        ibm_problem->set_dt(ibm_problem->_dt * dt_ratio); // dt = dt * dt_ratio
        nonlinear_result = ns.Solve(ibm_problem, xn_new, nullptr);

        if (!nonlinear_result.first) {
            LOG_SCOPE_F(WARNING, "Noninear solver failed.");
            dt_ratio = 0.5;
        } else {
            LOG_SCOPE_F(WARNING, "Noninear solver succeed.");
            *xn_1 = *xn;
            *xn   = *xn_new;
            ibm_problem->advance(*xn_new);
            ibm_problem->_solid_solver->record_boundary_points();
            LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first,
                  nonlinear_result.second.second);

            if (nonlinear_result.second.second < 20) {
                dt_ratio = 1.5;
            } else if (nonlinear_result.second.second > 25) {
                dt_ratio = 0.5;
            } else {
                dt_ratio = 1.0;
            }
        }
        CHECK_F(ibm_problem->_dt > 5e-6, "the time step is meaningless.");
    }
    LOG_F(WARNING, "finish time step.");

    return 1;
}

TEST_CASE("time_immersed_boundary_method2", "[long]") { REQUIRE(time_immersed_boundary_method2()); }