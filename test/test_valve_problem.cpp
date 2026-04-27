/**
 * @file test_valve_problem.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-05-26
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
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction2.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod2.h>
#include <PhysicsSolver/NavierStokesFlow/ProjectionSchemeGPU.h>
#include <PhysicsSolver/SolidSolver/ValveProblem/ValveSolver.h>
#include <catch.hpp>
#include <loguru/smtp.h>

using PressureTypegpu = GpuVector<double, double>;
using VelocityTypegpu = GpuVector<double, double3>;

using UserFluidSolver = pangu::ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu>;
using UserSolidSolver = dolfin::ValveSolver;

double current_pressure(double t) {
    // data
    static const double pressure_list[]
        = {0.0000,   9.3049,   9.3893,   9.4737,   10.2256,  10.9774,  14.8320,  19.0662,  25.0055,  31.1215,  38.8277,
           46.5339,  53.3130,  59.9917,  65.7018,  71.4118,  77.0797,  81.7119,  83.7607,  85.8094,  88.5115,  91.2635,
           94.2425,  97.2215,  100.0674, 102.8751, 105.3496, 107.8242, 109.5297, 111.1929, 112.7385, 114.2841, 115.4339,
           116.5705, 117.6868, 118.7922, 119.4247, 120.0573, 120.5338, 120.9885, 121.2837, 121.5789, 121.7898, 122.0007,
           122.1034, 122.1895, 122.1052, 122.0209, 121.7006, 121.3613, 120.9396, 120.5144, 119.9182, 119.3220, 118.6711,
           118.0021, 117.1504, 116.2989, 115.4555, 114.6120, 113.6748, 112.7193, 111.6228, 110.5263, 109.1637, 107.8011,
           106.3007, 104.7965, 103.3487, 101.8927, 100.0371, 98.1815,  95.7399,  93.2107,  90.2299,  87.1879,  82.6742,
           78.1605,  73.0977,  68.0237,  61.0248,  53.9030,  46.8594,  39.9931,  34.6611,  29.3732,  24.7743,  20.1754,
           16.3914,  12.6075,  10.4939,  8.4500,   6.6327,   4.8651,   3.6842,   2.5033,   1.8223,   1.1691,   0.5786,
           -0.0051,  -0.2581,  -0.5111,  -0.7036,  -0.8772,  -0.8772,  -0.8772,  -0.7085,  -0.5398,  -0.4395,  -0.3509,
           -0.3509,  -0.3509,  -0.1680,  0.0169,   0.1856,   0.3543,   0.5230,   0.6917,   0.7904,   0.8907,   1.0594,
           1.2281,   1.3968,   1.5654,   1.6648,   1.7645,   2.0175,   2.2706,   2.4618,   2.6485,   2.8172,   2.9859,
           3.1545,   3.3232,   3.5088,   3.6994,   3.9524,   4.2054,   4.3758,   4.5445,   4.7312,   4.9190,   5.0877,
           5.2565,   5.4423,   5.6275,   5.7962,   5.9649,   6.2179,   6.4710,   6.7515,   7.0226,   7.1069,   7.1913,
           7.4673,   7.7277,   7.8121,   8.0904,   8.5965,   8.9339,   9.2712,   9.3841,   9.4804,   9.6492,   9.8179,
           9.7340,   9.6491,   9.6491,   9.6491,   9.6491,   9.6491,   9.4718,   9.3049,   9.3893,   9.4737,   10.2256,
           10.9774,  14.8320,  19.0662,  25.0055,  31.1215,  38.8277,  46.5339,  53.3130,  59.9917,  65.7018,  71.4118,
           77.0797,  81.7119,  83.7607,  85.8094,  88.5115,  91.2635,  94.2425,  97.2215,  100.0674, 102.8751, 105.3496,
           107.8242, 109.5297, 111.1929, 112.7385, 114.2841, 115.4339, 116.5705, 117.6868, 118.7922, 119.4247, 120.0573,
           120.5338, 120.9885, 121.2837, 121.5789, 121.7898, 122.0007, 122.1034, 122.1895, 122.1052, 122.0209, 121.7006,
           121.3613, 120.9396, 120.5144, 119.9182, 119.3220, 118.6711, 118.0021, 117.1504, 116.2989, 115.4555, 114.6120,
           113.6748, 112.7193, 111.6228, 110.5263, 109.1637, 107.8011, 106.3007, 104.7965, 103.3487, 101.8927, 100.0371,
           98.1815,  95.7399,  93.2107,  90.2299,  87.1879,  82.6742,  78.1605,  73.0977,  68.0237,  61.0248,  53.9030,
           46.8594,  39.9931,  34.6611,  29.3732,  24.7743,  20.1754,  16.3914,  12.6075,  10.4939,  8.4500,   6.6327,
           4.8651,   3.6842,   2.5033,   1.8223,   1.1691,   0.5786,   -0.0051,  -0.2581,  -0.5111,  -0.7036,  -0.8772,
           -0.8772,  -0.8772,  -0.7085,  -0.5398,  -0.4395,  -0.3509,  -0.3509,  -0.3509,  -0.1680,  0.0169,   0.1856,
           0.3543,   0.5230,   0.6917,   0.7904,   0.8907,   1.0594,   1.2281,   1.3968,   1.5654,   1.6648,   1.7645,
           2.0175,   2.2706,   2.4618,   2.6485,   2.8172,   2.9859,   3.1545,   3.3232,   3.5088,   3.6994,   3.9524,
           4.2054,   4.3758,   4.5445,   4.7312,   4.9190,   5.0877,   5.2565,   5.4423,   5.6275,   5.7962,   5.9649,
           6.2179,   6.4710,   6.7515,   7.0226,   7.1069,   7.1913,   7.4673,   7.7277,   7.8121};
    static const int    pressure_num = 328;
    static const double dt           = 0.005;

    // calculation
    int n = t / dt;
    CHECK_F(n <= 326, "out of time range.");
    double k      = (t - n) / dt;
    double result = (pressure_list[n + 1] - pressure_list[n]) * k;
}

int time_immersed_boundary_method2() {
    loguru::add_file("test_EL_interactor_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_EL_interactor_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    double dt = 1e-2;
    double nu = 0.04;

    // Define meshes.
    std::cout << "Reading background mesh and immersed mesh.\n";
    double  rho         = 1.0;
    int3    dim_bg      = {129, 129, 129};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {13.0, 13.0, 13.0};
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
        = std::make_shared<ImmersedBoundaryMethod2<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
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
        // smtp::email_notification("mapengfei@mail.nwpu.edu.cn",
        //                          std::string(" first order element with lumping mass matrix. \n\n")
        //                              + std::string("special constraint on base.")
        //                              + "current time : " + std::to_string(ibm_problem->_t) + std::string("\n")
        //                              + "delta t : " + std::to_string(ibm_problem->_dt) + std::string("\n"));

        IBTimer timer("nonlinear test.");
        xn_new->axpy(-(1.0 + dt_ratio) / dt_ratio, *xn, *xn_1);
        xn_new->axpy(-1.0 - dt_ratio, *xn_new, *xn_new);

        ibm_problem->set_dt(ibm_problem->_dt * dt_ratio); // dt = dt * dt_ratio
        nonlinear_result = ns.Solve(ibm_problem, xn_new, nullptr);

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

TEST_CASE("time_immersed_boundary_method2", "[long]") { REQUIRE(time_immersed_boundary_method2()); }