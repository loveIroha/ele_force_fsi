/**
 * @file test_stokesflow.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

// Test Dirichlet pressure boundary conditions and Neumann velocity conditions.
// Modified by Ma Pengfei on 2022-04-07

#include <PhysicsSolver/NavierStokesFlow/ProjectionSchemeGPU.h>
#include <catch.hpp>
#include <io/writeVTK.h>
#include <loguru/IBTimer.h>

using namespace pangu;

int test_stokesflow_gpu() {
    loguru::add_file("test_stokesflow_gpu_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("test_stokesflow_gpu_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    using PressureType = StdVector<double, double>;
    using VelocityType = StdVector<double, double3>;

    using PressureTypegpu = GpuVector<double, double>;
    using VelocityTypegpu = GpuVector<double, double3>;

    int     n   = 65;
    double  dt  = 0.01;
    double  mu  = 0.1;
    double  rho = 1.0;
    int3    dim = make_int3(n, n, n);
    double3 dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    double3              origin = {0.0, 0.0, 0.0};
    std::vector<double3> data(dim.x * dim.y * dim.z);

    VelocityType u_b(dim);

    // TODO : set boundary velocity
    for (int i = 1; i < dim.x - 1; i++)
        for (int j = 1; j < dim.y - 1; j++) {
            u_b.data()[i + j * dim.x] = make_double3(1, 0, 0);
        }
    ProjectionSchemeGPU<PressureTypegpu, VelocityTypegpu> projection_scheme_gpu(dim, dh, dt, mu, rho);

    VelocityTypegpu u0(u_b);
    VelocityTypegpu fn(dim);

    // testing running.
    for (size_t i = 0; i < 2; i++) {
        projection_scheme_gpu._dt = dt;
        projection_scheme_gpu._t += dt;
        IBTimer time("Calculating...");
        u0 = projection_scheme_gpu.solveOneStep(fn, u0);
    }

    projection_scheme_gpu.record();
    return 1;
}

TEST_CASE("test_stokesflow_gpu", "[long]") { REQUIRE(test_stokesflow_gpu()); }