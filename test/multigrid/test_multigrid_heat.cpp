#include <PhysicsSolver/multigrid/HeatEquation1D.h>
#include <PhysicsSolver/multigrid/HeatEquation2D.h>
#include <PhysicsSolver/multigrid/HeatEquation3D.h>
#include <PhysicsSolver/multigrid/set_boundary_types.h>
#include <catch.hpp>

using namespace pangu;
using BCVectorType = StdVector<char, char>;
using VecType      = StdVector<double, double3>;

bool test_heat_1d() {
    // Define the mesh.
    int3    dim = make_int3(8193, 1, 1);
    double3 dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    // Define the boundary type.
    BCVectorType boundary_type(dim);
    boundary_type.data()[0]         = NEUMANN;
    boundary_type.data()[dim.x - 1] = NEUMANN;

    // Define time step.
    double dt = 0.5;
    double t  = 0.0;

    HeatEquation1D<VecType, BCVectorType> cpu_mgb(dim, dh, dt);

    VecType b(dim);
    VecType r(dim);
    VecType un(dim);
    VecType u_exact(dim);
    VecType u_old(dim);

    // Calculate boundary conditions on every layer.
    cpu_mgb.calculate_bcs(boundary_type);
    cpu_mgb.compute_b(b, un, t);
    cpu_mgb.compute_exact(un, t);

    // Set multigrid solver parameters.
    cpu_mgb.num_presmooth   = 1;
    cpu_mgb.num_aftersmooth = 1;

    // Time integration
    for (size_t i = 0; i < 1 / dt; i++) {
        t          = t + dt;         // t = 0.1
        cpu_mgb._t = t;              //
        cpu_mgb.compute_b(b, un, t); // b = f + un/dt
        cpu_mgb.compute_exact(u_exact, t);

        // Multigrid iteration
        for (size_t j = 0; j < 10; j++) {
            u_old = un;
            cpu_mgb.iterate(un, b, 0);
            cpu_mgb.compute_residuals(un, b, r, boundary_type, dim, dh);
            LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", j, r.inner(r) / r.size());
            r.axpy(-1.0, un, u_old);
            LOG_F(INFO, "after %ld th iteration, the reduce is      :  %e!", j, r.inner(r) / r.size());
            r.axpy(-1.0, un, u_exact);
            LOG_F(INFO, "after %ld th iteration, the error is       :  %e!", j, r.inner(r) / r.size());
        }
    }
    return true;
}

bool test_heat_2d() {
    // Define the mesh.
    int     n   = 257;
    int3    dim = make_int3(n, n, 1);
    double3 dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    // Define the boundary type.
    char         BCtop    = DIRICHLET;
    char         BCbottom = DIRICHLET;
    char         BCleft   = DIRICHLET;
    char         BCright  = DIRICHLET;
    BCVectorType boundary_type(dim);
    for (int i = 0; i < dim.x; i++) {
        boundary_type.data()[i + 0 * dim.x] = BCbottom;
    }
    for (int i = 0; i < dim.x; i++) {
        boundary_type.data()[i + (dim.y - 1) * dim.x] = BCtop;
    }
    for (int i = 0; i < dim.y; i++) {
        boundary_type.data()[0 + i * dim.x] = BCleft;
    }
    for (int i = 0; i < dim.y; i++) {
        boundary_type.data()[dim.x - 1 + i * dim.x] = BCright;
    }
    boundary_type.data()[0]                               = DIRICHLET;
    boundary_type.data()[dim.x - 1]                       = DIRICHLET;
    boundary_type.data()[(dim.x - 1) * dim.y]             = DIRICHLET;
    boundary_type.data()[(dim.x - 1) * dim.y + dim.x - 1] = DIRICHLET;

    // Define time step.
    double dt = 0.1;
    double t  = 0.0;

    // Define the multigrid solver.
    HeatEquation2D<VecType, BCVectorType> cpu_mgb(dim, dh, dt);

    VecType b(dim);
    VecType r(dim);
    VecType un(dim);
    VecType u_exact(dim);
    VecType u_old(dim);

    // Calculate boundary conditions on every layer.
    cpu_mgb.calculate_bcs(boundary_type);

    // Set multigrid solver parameters.
    cpu_mgb.num_presmooth   = 1;
    cpu_mgb.num_aftersmooth = 1;

    // Time integration
    for (size_t i = 0; i < 1 / dt; i++) {
        t          = t + dt;         // t = 0.1
        cpu_mgb._t = t;              //
        cpu_mgb.compute_b(b, un, t); // b = f + un/dt
        cpu_mgb.compute_exact(u_exact, t);
        cpu_mgb.apply_dirichlet_bcs(un, u_exact, boundary_type);

        for (size_t j = 0; j < 10; j++) {
            u_old = un;
            cpu_mgb.iterate(un, b, 0);
            r = 0;
            r.axpy(-1.0, un, u_exact);
            LOG_F(INFO, "after %ld th iteration, the error is       :  %e!", j, r.inner(r) / r.size());
            r = 0;
            r.axpy(-1.0, un, u_old);
            LOG_F(INFO, "after %ld th iteration, the reduce is      :  %e!", j, r.inner(r) / r.size());
            r = 0;
            cpu_mgb.compute_residuals(un, b, r, boundary_type, dim, dh);
            LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", j, r.inner(r) / r.size());
        }
    }
    return true;
}

bool test_heat_3d() {
    // Define the mesh.
    int     size = 257;
    int3    dim  = make_int3(size, size, size);
    double3 dh   = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    // Define the boundary type.
    BCVectorType boundary_type(dim);
    VecType      trash_1(dim);
    init_boundary_conditions(trash_1, boundary_type, dim);

    // Define time step.
    double dt = 0.1;
    double t  = 0.0;

    // Define the multigrid solver.
    HeatEquation3D<VecType, BCVectorType> cpu_mgb(dim, dh, dt);

    // Define variables
    VecType b(dim);
    VecType r(dim);
    VecType un(dim);
    VecType u_exact(dim);
    VecType u_old(dim);

    // Calculate boundary conditions on every layer.
    cpu_mgb.calculate_bcs(boundary_type);

    // // Set multigrid solver parameters.
    // cpu_mgb.num_presmooth = 2;
    // cpu_mgb.num_aftersmooth = 2;
    // cpu_mgb.num_exactsmooth = 100;

    // Time integration
    for (size_t i = 0; i < 1; i++) {
        t          = t + dt;         // t = 0.1
        cpu_mgb._t = t;              //
        cpu_mgb.compute_b(b, un, t); // b = f + un/dt
        cpu_mgb.compute_exact(u_exact, t);
        cpu_mgb.apply_dirichlet_bcs(un, u_exact, boundary_type);

        for (size_t j = 0; j < 10; j++) {
            u_old = un;
            cpu_mgb.iterate(un, b, 0);

            r.axpy(-1.0, un, u_exact);
            LOG_F(INFO, "after %ld th iteration, the error is       :  %e!", j, r.inner(r) / r.size());
            r.axpy(-1.0, un, u_old);
            LOG_F(INFO, "after %ld th iteration, the reduce is      :  %e!", j, r.inner(r) / r.size());
            cpu_mgb.compute_residuals(un, b, r, boundary_type, dim, dh);
            LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", j, r.inner(r) / r.size());
        }
    }
    return true;
}
TEST_CASE("test heat problems", "[long]") {
    // REQUIRE(test_heat_1d());
    // REQUIRE(test_heat_2d());
    REQUIRE(test_heat_3d());
}