#include <PhysicsSolver/multigrid/PoissonProblem3D.h>
#include <PhysicsSolver/multigrid/PoissonProblem3Dgpu.h>
#include <catch.hpp>
#include <loguru/IBTimer.h>
using namespace pangu;

using BCVectorType = StdVector<char, char>;
using VectorType   = StdVector<double, double>;

bool test_poisson_3d_gpu() {
    using gpu_vectype = GpuVector<double, double3>;
    using gpu_bctype  = GpuVector<char, char>;
    using vectype     = StdVector<double, double3>;
    using bctype      = StdVector<char, char>;

    int     size = 257;
    int3    dim  = make_int3(size, size, size);
    double3 dh   = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};

    PoissonProblem3D<vectype, BCVectorType> cpu_mgb(dim, dh);
    PoissonProblem3Dgpu<gpu_vectype>        gpu_mgb(dim, dh);

    vectype x(dim);
    vectype b(dim);
    vectype r(dim);
    vectype x_old(dim);
    vectype x_exact(dim);
    bctype  boundary_type(dim);

    cpu_mgb.compute_b(b);
    cpu_mgb.compute_exact(x_exact);
    init_boundary_conditions(x, boundary_type, dim);
    cpu_mgb.calculate_bcs(boundary_type);
    cpu_mgb.apply_dirichlet_bcs(x, x_exact);

    gpu_vectype gpu_x(x);
    gpu_vectype gpu_b(b);
    gpu_vectype gpu_r(r);
    gpu_vectype gpu_x_exact(x_exact);
    gpu_vectype gpu_x_old(x_old);
    gpu_bctype  gpu_boundary_type(boundary_type);

    // cpu_mgb.print_bcs(cpu_mgb.multi_bc[0],cpu_mgb.level_dim[0]);

    for (int i = 0; i < cpu_mgb.num_levels; i++) {
        gpu_mgb.multi_bc[i] = cpu_mgb.multi_bc[i];
    }

    // gpu_mgb.smooth(gpu_x, gpu_b, dim, dh, 1000);
    for (size_t i = 0; i < 100; i++) {
        LOG_F(INFO, "Step: %ld.", i);
        gpu_x_old = gpu_x;

        gpu_mgb.iterate(gpu_x, gpu_b, 0);
        gpu_mgb.compute_residuals(gpu_x, gpu_b, gpu_r, gpu_boundary_type, dim, dh);
        LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", i, gpu_r.inner(gpu_r) / gpu_r.size());
        gpu_r.axpy(-1.0, gpu_x, gpu_x_old);
        LOG_F(INFO, "after %ld th iteration, the convergence is :  %e!", i, gpu_r.inner(gpu_r) / gpu_r.size());
        gpu_r.axpy(-1.0, gpu_x, gpu_x_exact);
        LOG_F(INFO, "after %ld th iteration, the error is       :  %e!", i, gpu_r.inner(gpu_r) / gpu_r.size());
    }
    return true;
}

TEST_CASE("test poisson problems", "[long]") { REQUIRE(test_poisson_3d_gpu()); }