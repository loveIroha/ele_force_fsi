/// @date 2023-10-31
/// @file fsi_bar.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 直杆弯曲
///
///

#include <LBFGS.h>
#include <PhysicsSolver/StokesFlow3D/StaticFlow.h>
#include <config.h>
#include <io/parameters.h>
// #include <PhysicsSolver/SolidSolver/DiskDriven3D/BarSolver.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/BarProblem3D/BarSolver.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>

const int DIM = 3;



double3 function_location(const double3& x) { return make_double3(x.x, x.y, x.z); }

void calculate_position(std::vector<double4>& quadrature_rules_cur, const std::vector<double4>& quadrature_rules_ref,
                        const std::vector<double3>& displacement_qr) {
    size_t N = displacement_qr.size();
    CHECK_F(N == quadrature_rules_ref.size());
    CHECK_F(N == quadrature_rules_cur.size());
    // w分量为积分权重，禁止更改
    for (size_t i = 0; i < N; i++) {
        quadrature_rules_cur[i].x = quadrature_rules_ref[i].x + displacement_qr[i].x;
        quadrature_rules_cur[i].y = quadrature_rules_ref[i].y + displacement_qr[i].y;
        quadrature_rules_cur[i].z = quadrature_rules_ref[i].z + displacement_qr[i].z;
    }
}

class ImmersedBoundaryMethod {
  public:
    std::shared_ptr<BackgroundMesh3D<3>>          domain_mesh;
    std::shared_ptr<stokes_flow::StaticFlow<DIM>> fluid_solver;
    std::shared_ptr<ImmersedMesh>                 solid_mesh;
    std::shared_ptr<dolfin::BarSolver>            solid_solver;

    // 创建拉格朗日变量和欧拉变量的相互作用算子
    // 创建参考构型上的高斯积分法则
    // 将高斯积分点移动到当前构型上
    // 高斯积分点上的位移
    std::shared_ptr<MeshInteraction3D> mesh_interator_3D;
    std::vector<double4>               quadrature_rules_ref;
    std::vector<double4>               quadrature_rules_cur;
    std::vector<double3>               displacement_qr;

    // 创建定义在拉格朗日网格上的物理场(dofs)
    std::vector<double3>& _solid_forces;
    std::vector<double3>& _solid_velocities;
    std::vector<double3>& _solid_displacement;
    std::vector<double3>& _solid_location_ref;

    std::vector<double> _solid_forces_2;
    std::vector<double> _solid_velocities_2;
    std::vector<double> _solid_displacement_2;

    // 创建定义在背景网格上的速度场
    std::vector<double> eulerian_velocity_u;
    std::vector<double> eulerian_velocity_v;
    std::vector<double> eulerian_velocity_w;

    // 创建定义在背景网格上的力场
    std::vector<double> eulerian_force_u;
    std::vector<double> eulerian_force_v;
    std::vector<double> eulerian_force_w;

    // Constructor
    ImmersedBoundaryMethod(std::shared_ptr<BackgroundMesh3D<3>>          _domain_mesh,
                           std::shared_ptr<stokes_flow::StaticFlow<DIM>> _fluid_solver,
                           std::shared_ptr<ImmersedMesh> _solid_mesh, std::shared_ptr<dolfin::BarSolver> _solid_solver)
        : domain_mesh(_domain_mesh), fluid_solver(_fluid_solver), solid_mesh(_solid_mesh), solid_solver(_solid_solver),
          mesh_interator_3D(std::make_shared<MeshInteraction3D>(solid_mesh, domain_mesh)),
          quadrature_rules_ref(solid_mesh->get_quadrature_rules()),
          quadrature_rules_cur(solid_mesh->get_quadrature_rules()), displacement_qr(quadrature_rules_ref.size()),
          _solid_forces(solid_mesh->create_function<double3>("forces")),
          _solid_velocities(solid_mesh->create_function<double3>("velocities")),
          _solid_displacement(solid_mesh->create_function<double3>("displacement")),
          _solid_location_ref(solid_mesh->create_function<double3>("_solid_location_ref")),
          _solid_forces_2(algebra::flatten<double3, double>(_solid_forces)),
          _solid_velocities_2(algebra::flatten<double3, double>(_solid_velocities)),
          _solid_displacement_2(algebra::flatten<double3, double>(_solid_displacement)),
          eulerian_velocity_u(domain_mesh->get_size_u()), eulerian_velocity_v(domain_mesh->get_size_v()),
          eulerian_velocity_w(domain_mesh->get_size_w()), eulerian_force_u(domain_mesh->get_size_u()),
          eulerian_force_v(domain_mesh->get_size_v()), eulerian_force_w(domain_mesh->get_size_w()) {
        solid_mesh->set_function("_solid_location_ref", function_location);
    }

    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        // 1. 将 x 转换成 std::vector<double3> 类型
        std::vector<double3> X(x.size() / 3);
        std::vector<double3> U(x.size() / 3);
        std::vector<double3> Y(x.size() / 3);
        for (size_t i = 0; i < X.size(); i++) {
            X[i].x = x[3 * i] - _solid_location_ref[i].x;
            X[i].y = x[3 * i + 1] - _solid_location_ref[i].y;
            X[i].z = x[3 * i + 2] - _solid_location_ref[i].z;
        }

        LOG_F(INFO, "norm of the displacement: %f", algebra::norm(X));

        // 2. 调用 calculate_solid_velocity(X, U);
        calculate_solid_velocity(X, U);

        // 3. Y = X - _solid_displacement - dt * U;
        for (size_t i = 0; i < X.size(); i++) {
            Y[i].x = X[i].x - _solid_displacement[i].x - fluid_solver->_dt * U[i].x;
            Y[i].y = X[i].y - _solid_displacement[i].y - fluid_solver->_dt * U[i].y;
            Y[i].z = X[i].z - _solid_displacement[i].z - fluid_solver->_dt * U[i].z;
        }

        // 4. 将 Y 转换回 `Eigen::VectorXd` 类型
        for (size_t i = 0; i < X.size(); i++) {
            grad[3 * i]     = Y[i].x;
            grad[3 * i + 1] = Y[i].y;
            grad[3 * i + 2] = Y[i].z;
        }

        // 5. 计算 Y 的二范数
        double fx = algebra::norm(Y);
        return fx;
    }

    void calculate_solid_velocity(const std::vector<double3>& X, std::vector<double3>& U) {
        // 计算固体力
        LOG_F(INFO, "Calculate the force of the solid.");
        _solid_displacement_2 = algebra::flatten<double3, double>(X);
        solid_solver->solveOneStep(_solid_forces_2, _solid_displacement_2);
        _solid_forces = algebra::ripple<double3, double>(_solid_forces_2);

        // 调用分布算子
        LOG_F(INFO, "Distribute the force of the solid to the Eulerian grid.");
        mesh_interator_3D->distribute_force(eulerian_force_u, eulerian_force_v, eulerian_force_w, _solid_forces,
                                            quadrature_rules_cur);

        // 调用流体求解器
        LOG_F(INFO, "Solve the fluid problem.");
        fluid_solver->set_source(eulerian_force_u, eulerian_force_v,
                                 eulerian_force_w); // 设置右端项
        fluid_solver->set_velocity(eulerian_velocity_u, eulerian_velocity_v,
                                   eulerian_velocity_w);       // 设置上一时刻速度场
        auto [uh, vh, wh, ph]         = fluid_solver->solve(); // 求解当前时刻速度场
        auto eulerian_velocity_u_star = algebra::flatten(uh);  // 获取当前时刻速度场
        auto eulerian_velocity_v_star = algebra::flatten(vh);
        auto eulerian_velocity_w_star = algebra::flatten(wh);

        // 调用插值算子
        LOG_F(INFO, "Interpolate the velocity of the fluid to the Lagrangian grid.");
        mesh_interator_3D->interpolate_velocity(U, eulerian_velocity_u_star, eulerian_velocity_v_star,
                                                eulerian_velocity_w_star, quadrature_rules_cur);
    }

    void fun_update_disp() {
        // 计算高斯积分点处的固体位移（更新延拓算子和插值算子）
        LOG_F(INFO, "Calculate the displacement of the solid at the Gauss point (update the extension operator and the "
                    "interpolation operator).");
        solid_mesh->polynomial->evaluate_quadrature_points<double3, double>(
            _solid_displacement.data(), solid_mesh->dofmaps.data(), displacement_qr.data(), solid_mesh->num_cells());
        calculate_position(quadrature_rules_cur, quadrature_rules_ref, displacement_qr);
    }
    void fun_one_step(int i) {
        // 设置时间步长
        double dt        = domain_mesh->_TT / domain_mesh->_Nt;
        double t         = i * dt;
        fluid_solver->_t = t;
        solid_solver->_t = t;

        // 计算高斯积分点处的固体位移（更新延拓算子和插值算子）
        LOG_F(INFO, "Calculate the displacement of the solid at the Gauss point (update the extension operator and the "
                    "interpolation operator).");
        solid_mesh->polynomial->evaluate_quadrature_points<double3, double>(
            _solid_displacement.data(), solid_mesh->dofmaps.data(), displacement_qr.data(), solid_mesh->num_cells());
        calculate_position(quadrature_rules_cur, quadrature_rules_ref, displacement_qr);

        // 计算固体速度
        calculate_solid_velocity(_solid_displacement, _solid_velocities);
        calculate_solid_velocity(_solid_displacement, _solid_velocities);

        // 设置流体速度
        auto [un, vn, wn, pn] = fluid_solver->get_velocity_and_pressure(); // 求解当前时刻速度场
        eulerian_velocity_u   = algebra::flatten(un);                      // 获取当前时刻速度场
        eulerian_velocity_v   = algebra::flatten(vn);
        eulerian_velocity_w   = algebra::flatten(wn);

        // 计算固体位移
        LOG_F(INFO, "Calculate the displacement of the solid.");
        algebra::axpy(dt, _solid_velocities, _solid_displacement);

        // 输出结果
        if (i % (domain_mesh->_Nt / 1000) == 0) {
            LOG_F(INFO, "Record the results.");
            solid_solver->record<double3, double>(_solid_forces, _solid_displacement, t);
            auto [f1, f2, f3]     = fluid_solver->get_source();
            auto [un, vn, wn, pn] = fluid_solver->get_velocity_and_pressure();
            fluid_solver->record(un, vn, wn, f1, f2, f3, pn, t);
        }
    }

    void fun() {
        for (int i = 1; i < domain_mesh->_Nt + 1; i++) {
            fun_one_step(i);
        }
    }
};

template <typename T1, typename T2>
T2 convert_array(T1 t1) {
    T2 t2;
    return t2;
}

using Eigen::VectorXd;
using namespace LBFGSpp;

void fsi_lid_simulation(std::shared_ptr<BackgroundMesh3D<3>>          domain_mesh,
                        std::shared_ptr<stokes_flow::StaticFlow<DIM>> fluid_solver,
                        std::shared_ptr<ImmersedMesh> solid_mesh, std::shared_ptr<dolfin::BarSolver> solid_solver) {
    // 创建流固耦合对象
    auto ibm = std::make_shared<ImmersedBoundaryMethod>(domain_mesh, fluid_solver, solid_mesh, solid_solver);

    for (int i = 1; i < ibm->domain_mesh->_Nt + 1; i++) {
        // 设置时间步长
        double dt             = ibm->domain_mesh->_TT / ibm->domain_mesh->_Nt;
        double t              = i * dt;
        ibm->fluid_solver->_t = t;
        ibm->solid_solver->_t = t;

        // 插值算子和延拓算子
        LOG_F(INFO, "Calculate the displacement of the solid at the Gauss point (update the extension operator and the "
                    "interpolation operator).");
        ibm->fun_update_disp();

        // 计算位移
        //     1. 显格式：上一时刻的位移
        //     2. 隐格式：通过牛顿迭代方法计算当前时刻的位移
        // 初值，将 ibm->_solid_displacement 转换成 Eigen::VectorXd 类型
        double   fx;
        VectorXd x = VectorXd::Zero(ibm->_solid_displacement.size() * 3);
        for (size_t j = 0; j < ibm->_solid_displacement.size(); j++) {
            x[3 * j]     = ibm->_solid_displacement[j].x + ibm->_solid_location_ref[j].x;
            x[3 * j + 1] = ibm->_solid_displacement[j].y + ibm->_solid_location_ref[j].y;
            x[3 * j + 2] = ibm->_solid_displacement[j].z + ibm->_solid_location_ref[j].z;
        }

        // 创建 LBFGS 求解器
        LBFGSParam<double> param;
        // param.max_step = 1e-4;
        param.epsilon        = 1e-3;
        param.max_iterations = 100;
        // param.ftol = 0.1;

        // 设置线搜索方式
        // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
        // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        // param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

        // LBFGSSolver<double, LineSearchBracketing> solver(param);
        // LBFGSSolver<double, LineSearchBacktracking> solver(param);
        LBFGSSolver<double, LineSearchNocedalWright> solver(param);
        // LBFGSSolver<double, LineSearchMoreThuente> solver(param);
        // LBFGSSolver<double> solver(param);
        int niter = solver.minimize(*ibm, x, fx);
        LOG_F(INFO, "LBFGSolver : %d iterations, %f error.", niter, fx);

        // 将 x 取出，放到 ibm->_solid_displacement 中
        for (size_t j = 0; j < ibm->_solid_displacement.size(); j++) {
            ibm->_solid_displacement[j].x = x[3 * j] - ibm->_solid_location_ref[j].x;
            ibm->_solid_displacement[j].y = x[3 * j + 1] - ibm->_solid_location_ref[j].y;
            ibm->_solid_displacement[j].z = x[3 * j + 2] - ibm->_solid_location_ref[j].z;
        }

        // 计算速度
        ibm->calculate_solid_velocity(ibm->_solid_displacement, ibm->_solid_velocities);

        // 设置流体速度
        auto [un, vn, wn, pn]    = ibm->fluid_solver->get_velocity_and_pressure(); // 求解当前时刻速度场
        ibm->eulerian_velocity_u = algebra::flatten(un);                           // 获取当前时刻速度场
        ibm->eulerian_velocity_v = algebra::flatten(vn);
        ibm->eulerian_velocity_w = algebra::flatten(wn);

        // 计算固体位移
        LOG_F(INFO, "Calculate the displacement of the solid.");
        algebra::axpy(dt, ibm->_solid_velocities, ibm->_solid_displacement);

        // 输出结果
        {
            LOG_F(INFO, "Record the results.");
            ibm->solid_solver->record<double3, double>(ibm->_solid_forces, ibm->_solid_displacement, t);
            auto [f1, f2, f3]     = ibm->fluid_solver->get_source();
            auto [un, vn, wn, pn] = ibm->fluid_solver->get_velocity_and_pressure();
            ibm->fluid_solver->record(un, vn, wn, f1, f2, f3, pn, t);
        }
    }

    // ibm->fun();
}

int fsi_simulation(std::shared_ptr<dolfin::Mesh>                      solid_mesh_dolfin,
                   std::shared_ptr<dolfin::MeshFunction<std::size_t>> solid_mesh_boundary, int Ns, int Nt, double T,
                   double mu_f) {
    ScopeProfiler        _{__func__};
    int                  Nf = 32;
    int                  Nx = Nf;
    int                  Ny = Nf;
    int                  Nz = Nf;
    std::array<int, DIM> dim_bg{Nx, Ny, Nz};

    double                  Lx = 2.0;
    double                  Ly = 2.0;
    double                  Lz = 2.0;
    std::array<double, DIM> L{Lx, Ly, Lz};

    double                             rho = 1.0;
    std::map<std::string, std::string> paths;
    std::ostringstream                 path;
    path << "BarBending_3D_implicit_" << Ns << "_" << Nt << "_" << T << "_" << mu_f << "/";

    paths["output_file"] = path.str();

    loguru::add_file((path.str() + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file((path.str() + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    // 创建背景网格
    auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(Nt, dim_bg, L, T, rho, mu_f);

    // 创建流体求解器
    auto fluid_solver
        = std::make_shared<stokes_flow::StaticFlow<DIM>>(Nt, dim_bg, L, T, rho, mu_f, paths["output_file"]);

    // 创建固体网格
    auto solid_mesh = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);

    // 创建固体求解器
    auto solid_solver
        = std::make_shared<dolfin::BarSolver>(solid_mesh->get_dolfin_mesh(), solid_mesh_boundary, paths["output_file"]);

    fsi_lid_simulation(domain_mesh, fluid_solver, solid_mesh, solid_solver);

    printScopeProfiler();
    return 0;
}

void bar_bending(int Ns, int Nt, double T, double mu_f) {
    std::map<std::string, std::string> paths;
    auto                               mesh_file = geometry_path("temp/bar_");
    mesh_file += std::to_string(Ns);
    mesh_file += ".xdmf";
    paths["mesh_file"] = mesh_file;

    // auto domains_file = geometry_path("temp/bar_behind_cylinder_");
    // domains_file += std::to_string(Ns);
    // domains_file += "_domains.xml";
    // paths["domains_file"] = domains_file;

    auto boundaries_file = geometry_path("temp/bar_");
    boundaries_file += std::to_string(Ns);
    boundaries_file += "_boundary.xml";
    paths["boundaries_file"] = boundaries_file;

    // // 读取固体网格
    // auto mesh_file       = paths["mesh_file"];
    // auto domains_file    = paths["domains_file"];
    // auto boundaries_file = paths["boundaries_file"];
    // auto output_file     = paths["output_file"];

    // 读取网格文件
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();

    // auto materal_types =
    // std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin,
    // domains_file);
    auto solid_mesh_boundary = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, boundaries_file);

    // 运行算例
    fsi_simulation(solid_mesh_dolfin, solid_mesh_boundary, Ns, Nt, T, mu_f);
    printScopeProfiler();
}

bool help(int argc, char* argv[]) {
    if (parameters::get_arg(argv, argv + argc, "--help")) {
        printf("直杆弯曲程序流固耦合模拟。使用方法: \n");
        printf("    程序名\t\t\t\t-参数名\t参数值\n");
        printf("    ./fsi_bar_bending\t-Nt\t10000\n");
        printf("\n");
        printf("可选参数:\n");
        printf("--help\t\t提供程序运行帮助\n");
        printf("-Nt\t\t时间步数\n");
        printf("-Ns\t\t固体网格密度\n");
        printf("-T\t\t时间长度\n");
        printf("-mu_f\t\t流体粘性系数\n");
        return true;
    } else {
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (help(argc, argv)) return 0;

    double T    = parameters::get_argval<double>(argv, argv + argc, "-T", 10);
    double mu_f = parameters::get_argval<double>(argv, argv + argc, "-mu_f", 0.04);
    int    Ns   = parameters::get_argval<int>(argv, argv + argc, "-Ns", 40);
    int    Nt   = parameters::get_argval<int>(argv, argv + argc, "-Nt", 20000);
    bar_bending(Ns, Nt, T, mu_f);
    return 0;
}