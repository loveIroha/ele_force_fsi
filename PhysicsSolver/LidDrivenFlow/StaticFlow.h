/// @date 2023-04-12
/// @file StaticFlow.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief Static Flow 静态流体，速度为零Neumann边界条件，压强全为零Dirichlet边界条件。
///

#include <PhysicsSolver/StokesFlow/PressureSolver.h>
#include <PhysicsSolver/StokesFlow/PressureSolverGPU.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>
#include <PhysicsSolver/StokesFlow/TentitiveVelocity.h>
#include <PhysicsSolver/StokesFlow/TentitiveVelocityGPU.h>
#include <PhysicsSolver/multigrid/set_boundary_types.h>
#include <io/IBTimer.h>

using namespace pangu;

// 定义数据类型
using PressureType         = StdVector<double, double>;
using VelocityType         = StdVector<double, double3>;
using PressureBoundaryType = StdVector<char, char>;
using VelocityBoundaryType = StdVector<char, char>;

// CUDA数据类型
using GPUPressureType         = GpuVector<double, double>;
using GPUVelocityType         = GpuVector<double, double3>;
using GPUPressureBoundaryType = GpuVector<char, char>;
using GPUVelocityBoundaryType = GpuVector<char, char>;

using FluidSolver = ProjectionScheme<PressureType, VelocityType, PressureBoundaryType, VelocityBoundaryType>;
using GPUFluidSolver
    = ProjectionSchemeGPU<GPUPressureType, GPUVelocityType, GPUPressureBoundaryType, GPUVelocityBoundaryType>;

class StaticFlow {
  public:
    VelocityType         f;
    VelocityType         u0;
    PressureType         p0;
    PressureBoundaryType pbcs;
    VelocityBoundaryType vbcs;

    GPUVelocityType         gpu_f;
    GPUVelocityType         gpu_u0;
    GPUPressureType         gpu_p0;
    GPUPressureBoundaryType gpu_pbcs;
    GPUVelocityBoundaryType gpu_vbcs;

    FluidSolver    projection_scheme;
    GPUFluidSolver gpu_projection_scheme;

  public:
    StaticFlow(int3 dim, double3 dh, double dt, double nu, double rho,
               std::string velocity_file = "fluid/velocity/data.pvd",
               std::string pressure_file = "fluid/pressure/data.pvd", std::string force_file = "fluid/force/data.pvd")
        : f(dim), u0(dim), p0(dim), pbcs(dim), vbcs(dim), gpu_f(dim), gpu_u0(dim), gpu_p0(dim), gpu_pbcs(dim),
          gpu_vbcs(dim), projection_scheme(dim, dh, dt, nu, rho),
          gpu_projection_scheme(dim, dh, dt, nu, rho, velocity_file, pressure_file, force_file) {
        init_boundary_conditions(u0, vbcs, dim, NEUMANN, double3{0, 0, 0}, NEUMANN, double3{0, 0, 0}, NEUMANN,
                                 double3{0, 0, 0}, NEUMANN, double3{0, 0, 0}, NEUMANN, double3{0, 0, 0}, NEUMANN,
                                 double3{0, 0, 0});

        init_boundary_conditions(p0, pbcs, dim, DIRICHLET, 0.0, DIRICHLET, 0.0, DIRICHLET, 0.0, DIRICHLET, 0.0,
                                 DIRICHLET, 0.0, DIRICHLET, 0.0);

        // 将边界类型和边界值复制到GPU变量中
        gpu_u0   = u0;
        gpu_p0   = p0;
        gpu_pbcs = pbcs;
        gpu_vbcs = vbcs;

        // 将边界条件复制到流体求解器中
        projection_scheme.u     = u0;
        projection_scheme.p     = p0;
        gpu_projection_scheme.u = u0;
        gpu_projection_scheme.p = p0;

        // 处理边界条件
        projection_scheme.vbcs     = vbcs;
        projection_scheme.pbcs     = pbcs;
        gpu_projection_scheme.vbcs = vbcs;
        gpu_projection_scheme.pbcs = pbcs;

        // 生成并复制多重网格求解器的边界类型
        projection_scheme.tv->calculate_bcs(vbcs);
        projection_scheme.ps->calculate_bcs(pbcs);
        for (int i = 0; i < projection_scheme.tv->num_levels; i++) {
            gpu_projection_scheme.tv->multi_bc[i] = projection_scheme.tv->multi_bc[i];
            gpu_projection_scheme.ps->multi_bc[i] = projection_scheme.ps->multi_bc[i];
        }
    }

    ~StaticFlow() {}

    std::vector<double3> solveOneStep(const std::vector<double3>& fn, std::vector<double3>& un) {
        // 更新时间
        projection_scheme._t += projection_scheme._dt;
        gpu_projection_scheme._t += gpu_projection_scheme._dt;

        return gpu_projection_scheme.solveOneStep(fn, un);
    }

    void solveOneStep(const std::vector<double3>& vector_fn, const std::vector<double3>& vector_un,
                      std::vector<double3>& vector_u, std::vector<double>& vector_p) {
        gpu_projection_scheme.solveOneStep(vector_fn, vector_un, vector_u, vector_p);
    }

    void solveOneStep(const std::vector<double3>& vector_fn, const std::vector<double3>& vector_un,
                      const std::vector<double>& vector_sn, std::vector<double3>& vector_u,
                      std::vector<double>& vector_p) {
        gpu_projection_scheme.solveOneStep(vector_fn, vector_un, vector_sn, vector_u, vector_p);
    }

    double get_t() { return projection_scheme._t; }
    double get_dt() { return projection_scheme._dt; }
    double get_nu() { return projection_scheme._nu; }

    void set_t(double t) {
        gpu_projection_scheme.set_t(t);
        projection_scheme.set_t(t);
    }

    void set_dt(double dt) {
        gpu_projection_scheme.set_dt(dt);
        projection_scheme.set_dt(dt);
    }

    void record(const std::vector<double3>& forces, const std::vector<double3>& velocities,
                const std::vector<double>& pressures, double t) {
        gpu_projection_scheme.record(forces, velocities, pressures, t);
    }
};
