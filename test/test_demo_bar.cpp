/**
 * @file bar_problem.cpp
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
#include <PhysicsSolver/SolidSolver/BarProblem/BarSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <catch.hpp>
#include <loguru/smtp.h>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;

using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::BarSolver;

int bar_problem() {
    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    smtp::CSmtp email_qq(25, "smtp.qq.com", "499908174@qq.com", "sbiaoieuregscafh", "mapengfei@mail.nwpu.edu.cn",
                         "测试邮箱", "成功发送邮件！");

    double dt = 1e-2;
    double nu = 0.04;

    // Define meshes.
    std::cout << "Reading background mesh and immersed mesh.\n";
    auto    domain      = std::make_shared<mshr::Box>(dolfin::Point(0.1, 0.5, 0.5), dolfin::Point(0.9, 0.6, 0.6));
    auto    solid_mesh  = std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain, 40));
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
        // Send Email notifications.
        email_qq.content
            = std::string(" first order element with lumping mass matrix. \n\n")
              + std::string("special constraint on base.") + "current time : " + std::to_string(ibm_problem->_t)
              + std::string("\n") + "delta t : " + std::to_string(ibm_problem->_dt) + std::string("\n")
            = "第二封邮件";
        smtp::email_notification(email_qq) == 0;

        // Approximate the solution with linear extrapolation
        LOG_SCOPE_F(INFO, "nonlinear test.");
        xn_new->axpy(-(1.0 + dt_ratio) / dt_ratio, *xn, *xn_1);
        xn_new->axpy(-1.0 - dt_ratio, *xn_new, *xn_new);

        // Set the time step
        ibm_problem->set_dt(ibm_problem->_dt * dt_ratio); // dt = dt * dt_ratio
        nonlinear_result = ns.Solve(ibm_problem, xn_new, nullptr);

        // Call the nonlinear solver
        if (!nonlinear_result.first) {
            LOG_F(WARNING, "Noninear solver failed.");
            dt_ratio = 0.5;
        } else {
            LOG_F(WARNING, "Noninear solver succeed.");
            *xn_1 = *xn;
            *xn   = *xn_new;
            ibm_problem->advance(*xn_new);

            LOG_F(WARNING, "t : %.12e, dt : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, ibm_problem->_dt,
                  xn->_data[0].x, xn->_data[0].y, xn->_data[0].z);
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

    return 1;
}

TEST_CASE("bar_problem", "[long]") { REQUIRE(bar_problem()); }