/// @date 2023-03-20
/// @file test_ibm_bended_bar_implicit_penalty.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/LinearProblem.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>
#include <PhysicsSolver/LidDrivenFlow/StaticFlow.h>
#include <PhysicsSolver/SolidSolver/BarProblem/BarSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <PhysicsSolver/StokesFlow/ProjectionSchemeGPU.h>

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

using UserFluidSolver = StaticFlow;
using UserSolidSolver = dolfin::BarSolver;

std::shared_ptr<ImmersedMesh> read_mesh(std::string filename) {
    dolfin::XDMFFile mesh_file_1(filename.c_str());
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto solid_mesh = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);
    return solid_mesh;
}

int main(int argc, char* argv[]) {
    CHECK_F(argc == 5, "必须输入四个参数");

    // 固体网格大小、流体网格大小、时间比率、GPU设备

    // 离散参数
    double T               = 10.0;
    int    solid_mesh_size = std::stoi(argv[1]);
    int    fluid_mesh_size = std::stoi(argv[2]);
    int    time_step_ratio = std::stoi(argv[3]);
    int    time_steps      = T * fluid_mesh_size * time_step_ratio;
    double dt              = T / time_steps;

    // 设置GPU设备
    gpu::find_devices();
    gpu::set_device(std::stoi(argv[4]));

    // Define mesh for immersed body
    std::ostringstream solid_mesh_path;
    std::ostringstream solid_mesh_boundary_path;
    solid_mesh_path << "../geometry/temp/bar_" << solid_mesh_size << ".xdmf";
    solid_mesh_boundary_path << "../geometry/temp/bar_" << solid_mesh_size << "_boundary.xml";
    auto solid_mesh        = read_mesh(solid_mesh_path.str());
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();
    auto solid_mesh_boundaries
        = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, solid_mesh_boundary_path.str());

    // 输出数据路径
    std::ostringstream result_path;
    result_path << "demo_bar_implicit_" << fluid_mesh_size << "_" << solid_mesh_size << "_" << time_steps << "/";

    // 日志文件路径
    loguru::add_file((result_path.str() + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file((result_path.str() + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // Define mesh for background domain
    double  nu          = 0.04;
    double  rho         = 1.0;
    int3    dim_bg      = {fluid_mesh_size, fluid_mesh_size, fluid_mesh_size};
    double3 origin_bg   = {0.0, 0.0, 0.0};
    double3 box_size_bg = {2.0, 2.0, 2.0};
    auto    fluid_mesh  = std::make_shared<BackgroundMesh2>(dim_bg, origin_bg, box_size_bg, nu, rho);

    // Define solid solver and fluid solver
    auto fluid_solver = std::make_shared<UserFluidSolver>(
        fluid_mesh->dim, fluid_mesh->dh, dt, fluid_mesh->mu, fluid_mesh->rho,
        result_path.str() + "fluid/velocity/data.pvd", result_path.str() + "fluid/pressure/data.pvd",
        result_path.str() + "fluid/force/data.pvd");
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh_dolfin, solid_mesh_boundaries, result_path.str());

    // Define immersed boundary method solver
    auto ibm_problem
        = std::make_shared<ImmersedBoundaryMethod<UserSolidSolver, UserFluidSolver, StdVector<double, double3>>>(
            solid_mesh, fluid_mesh, solid_solver, fluid_solver);

    LOG_F(INFO, "Create Newton solver.");
    auto bicgstab = std::make_shared<BiCGSTAB<StdVector<double, double3>>>(ibm_problem->unkown_size() / 3);
    bicgstab->set_tolerance(1e-2);
    bicgstab->set_max_iteration(20);
    NewtonSolver<StdVector<double, double3>> ns(bicgstab);
    ns.max_nonlinear_tolerance = 1e-5;
    ns.max_nonlinear_iteration = 100;
    LOG_F(INFO, "Done.");

    // Initialize variables
    LOG_F(INFO, "Initialize variables.");
    auto b      = std::make_shared<StdVector<double, double3>>(); // x_{n-1}
    auto xn     = std::make_shared<StdVector<double, double3>>(); // x_{n}
    auto xn_1   = std::make_shared<StdVector<double, double3>>();
    auto xn_new = std::make_shared<StdVector<double, double3>>();
    b->resize(ibm_problem->unkown_size() / 3);
    xn->resize(ibm_problem->unkown_size() / 3);
    xn_1->resize(ibm_problem->unkown_size() / 3);
    xn_new->resize(ibm_problem->unkown_size() / 3);
    LOG_F(INFO, "Done.");

    ibm_problem->get_solid_positions(*xn);
    *xn_1   = *xn;
    *xn_new = *xn;

    double dt_ratio         = 1.0;
    auto   nonlinear_result = std::make_pair(true, std::make_pair(0.0, 0));
    while (ibm_problem->_t < T) {
        IBTimer timer("Nonlinear solver");
        // Initial approximation $\mathbf{X}_{h,n+1}^{(k,l)}$
        xn_new->axpy(-(1.0 + dt_ratio) / dt_ratio, *xn, *xn_1);
        xn_new->axpy(-1.0 - dt_ratio, *xn_new, *xn_new);
        // Set dt = dt * dt_ratio for solid, fluid and ibm
        ibm_problem->set_dt(ibm_problem->_dt * dt_ratio);
        // Call nonlinear solver
        nonlinear_result = ns.Solve(ibm_problem, xn_new, nullptr);
        // Analyse the result of nonlinear solver
        // If it is failed, reduce the time step
        if (!nonlinear_result.first) {
            LOG_F(WARNING, "Noninear solver failed.");
            dt_ratio = 0.5;
        }
        // If it is successful,
        else {
            LOG_F(WARNING, "Noninear solver succeed.");
            // Update the positons of solid
            *xn_1 = *xn;
            *xn   = *xn_new;
            ibm_problem->advance(*xn_new);

            LOG_F(WARNING, "t : %.12e, dt : %.12e, xn: %.12e, %.12e, %.12e", ibm_problem->_t, ibm_problem->_dt,
                  xn->_data[0].x, xn->_data[0].y, xn->_data[0].z);
            LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first,
                  nonlinear_result.second.second);
            // If the number of iteration is too large, enlarge the dt_ratio.
            if (nonlinear_result.second.second < 10) {
                dt_ratio = 1.2;
            } else if (nonlinear_result.second.second > 40) {
                dt_ratio = 0.5;
            } else {
                dt_ratio = 1.0;
            }
        }
        CHECK_F(ibm_problem->_dt > 5e-6, "the time step is meaningless.");
    }
    return 0;
}
