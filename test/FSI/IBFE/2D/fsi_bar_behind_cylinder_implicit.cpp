/// @date 2023-09-15
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 圆柱后面加一根弹性直杆
///
///
///      单位换算成cm, g, s三种基本单位后
///      1. 流入速度为 200 cm/s         2 m/s
///      2. 粘性      10  g/(cm*s)     1 kg/(m*s)
///      3. 密度      1   g/cm^3       1000 kg/m^3
///      4. G_s      5*10^6 g/(cm*s^2) 2*10^6 kg/(m*s^2)
///      5. nv       0.499             0.499
///      计算区域换算成 [0,246]×[0,41]    [0,2.46]×[0,0.41]

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <PhysicsSolver/SolidSolver/BarBehindCylinder2D/BarBehindCylinder.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <config.h>
#include <io/writeVTK.h>

#include <tuple>

const int    DEGREE = 1;
const int    DIM    = 2;
const double pi     = 3.14159265358979;

double2 left_boundary(double2 p) {
    double y = p.y;
    double u = 2 * 1.5 * (4.0 * y * (0.41 - y)) / (0.41 * 0.41);
    double v = 0.0;
    return {u, v};
}



double2 function_reference_configuration(const double3& x) { return make_double2(x.x, x.y); }
class PipeFlow {
  public:
    StokesFlow<DIM> stokes_flow;

    double  _t = 0.0;
    double& _dt;

    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> v;
    std::vector<std::vector<double>> p;
    std::vector<std::vector<double>> un;
    std::vector<std::vector<double>> vn;
    std::vector<std::vector<double>> uf;
    std::vector<std::vector<double>> vf;

    VTIWriter velocity_writer;
    VTIWriter pressure_writer;
    VTIWriter force_writer;

    PipeFlow(BackgroundMesh2D<2> mesh, std::string velocity_file = "fluid/velocity/data.pvd",
             std::string pressure_file = "fluid/pressure/data.pvd", std::string force_file = "fluid/force/data.pvd")
        : stokes_flow{mesh._Nt, {mesh._Nx, mesh._Ny}, {mesh._Lx, mesh._Ly}, mesh._T, mesh._rho, mesh._mu},
          _dt(stokes_flow._dt), velocity_writer{velocity_file}, pressure_writer{pressure_file}, force_writer{force_file}

    {
        // 设置边界类型
        int ubt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, NEUMANN};
        int vbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, NEUMANN};
        int pbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, DIRICHLET};

        double velocity_u[4] = {0, 0, 0, 0};
        double velocity_v[4] = {0, 0, 0, 0};
        double pressure[4]   = {0, 0, 0, 0};

        stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);

        set_boundary_values();

        int _Nx = stokes_flow._Nx;
        int _Ny = stokes_flow._Ny;

        u = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        v = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        p = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        un = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vn = make_vector_2D<double>(_Nx + 2, _Ny + 1);

        uf = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vf = make_vector_2D<double>(_Nx + 2, _Ny + 1);
    }

    void set_boundary_values() {
        printf("当前时刻为：%e\n", _t);
        for (size_t j = 1; j < stokes_flow._Ny + 1; j++) {
            double  param_t       = _t < 2.0 ? 0.5 * (1.0 - std::cos(0.5 * pi * _t)) : 1.0;
            double  y             = stokes_flow._dy * j - 0.5 * stokes_flow._dy;
            double2 result        = left_boundary({0.0, y});
            stokes_flow.ubv[0][j] = result.x * param_t;
        }
    }

    auto get_variables() {
        return std::make_tuple(std::ref(u), std::ref(v), std::ref(p), std::ref(un), std::ref(vn), std::ref(uf),
                               std::ref(vf));
    }
};

class ImmersedBoundaryMethod : public algebra::NonlinearProblem {
  public:
    BasicMesh<DEGREE, DIM>                     basic_mesh;
    PipeFlow                                   pipe_flow;
    std::shared_ptr<dolfin::BarBehindCylinder> flow_past_cylinder;
    BackgroundMesh2D<2>                        domain_mesh;

    // 在固体网格上定义函数
    // NOTE: 的变量会在calculate_velocity函数中被修改
    // NOTE: 非 TEMPORARY 的变量每个时间变量只会被修改一次
    std::vector<double2>& _solid_forces; // TEMPORARY
    std::vector<double2>& _solid_velocities;
    std::vector<double2>& _solid_displacement;
    std::vector<double2>& _solid_displacement_old;
    std::vector<double2>& _solid_reference_configuration;
    std::vector<double>   _solid_forces_2; // TEMPORARY
    std::vector<double>   _solid_velocities_2;
    std::vector<double>   _solid_displacement_2;
    std::vector<double>   _solid_displacement_old_2;
    std::vector<double4>  _eulerian_quadrature_rules;

    // 在流体网格上定义函数
    std::vector<double>  eulerian_force_u;    // TEMPORARY
    std::vector<double>  eulerian_force_v;    // TEMPORARY
    std::vector<double>  eulerian_velocity_u; // TEMPORARY
    std::vector<double>  eulerian_velocity_v; // TEMPORARY
    std::vector<double>& _fluid_pressures;

    double  t;
    double& dt;
    double& T;

    ImmersedBoundaryMethod(BasicMesh<DEGREE, DIM> _solid_mesh, BackgroundMesh2D<DIM> _domain_mesh, PipeFlow ns_solver,
                           std::shared_ptr<dolfin::BarBehindCylinder> solid_solver)
        : basic_mesh(_solid_mesh), pipe_flow(ns_solver), flow_past_cylinder(solid_solver), domain_mesh(_domain_mesh),
          _solid_forces(basic_mesh.create_function<double2>("forces")),
          _solid_velocities(basic_mesh.create_function<double2>("velocities")),
          _solid_displacement(basic_mesh.create_function<double2>("displacement")),
          _solid_displacement_old(basic_mesh.create_function<double2>("displacement_old")),
          _solid_reference_configuration(basic_mesh.create_function<double2>("reference_configuration")),
          _solid_forces_2(algebra::flatten<double2, double>(_solid_forces)),
          _solid_velocities_2(algebra::flatten<double2, double>(_solid_velocities)),
          _solid_displacement_2(algebra::flatten<double2, double>(_solid_displacement)),
          _solid_displacement_old_2(algebra::flatten<double2, double>(_solid_displacement_old)),
          _eulerian_quadrature_rules(basic_mesh.quadrature_rules), eulerian_force_u(domain_mesh.get_size_u()),
          eulerian_force_v(domain_mesh.get_size_v()), eulerian_velocity_u(domain_mesh.get_size_u()),
          eulerian_velocity_v(domain_mesh.get_size_v()),
          _fluid_pressures(domain_mesh.create_function<double>("pressures")), t(0.0), dt(pipe_flow.stokes_flow._dt),
          T(pipe_flow.stokes_flow._T) {
        basic_mesh.set_function("reference_configuration", function_reference_configuration);
        xk.resize(basic_mesh.num_dofs() * DIM);
        rk.resize(basic_mesh.num_dofs() * DIM);
    }

    void calculate_solid_velocity(const std::vector<double>& X, std::vector<double>& U) {
        static int count = 0;
        count++;
        LOG_F(INFO, "调用 calculate_solid_velocity 次数： %d", count);

        auto [u, v, p, un, vn, uf, vf] = pipe_flow.get_variables();

        // 1. 求解固体的力
        // 修改了 _solid_forces_2 和 _solid_forces 变量
        std::vector<double> _solid_forces_2(X.size());
        flow_past_cylinder->solveOneStep(_solid_forces_2, X);
        _solid_forces = algebra::ripple<double2, double>(_solid_forces_2);

        // 2. 固体力的延拓
        // 修改了 eulerian_force_u, eulerian_force_v 两个变量
        std::vector<double2> force_qr(_eulerian_quadrature_rules.size());
        basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
            _solid_forces.data(), basic_mesh.dofmaps.data(), force_qr.data(), basic_mesh.num_cells());
        basic_mesh.distribute_force(eulerian_force_u, eulerian_force_v, force_qr, _eulerian_quadrature_rules,
                                    make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_u(),
                                    domain_mesh.get_dim_v());

        // 3. 求解速度场
        // 修改了 pipe_flow 中的 uf, vf
        // 修改了 eulerian_velocity_u, eulerian_velocity_v 两个变量
        uf = algebra::ripple(eulerian_force_u, uf.size());
        vf = algebra::ripple(eulerian_force_v, vf.size());
        pipe_flow.set_boundary_values();
        pipe_flow.stokes_flow.solve_one_step(u, v, p, uf, vf, un, vn);
        eulerian_velocity_u = algebra::flatten(u);
        eulerian_velocity_v = algebra::flatten(v);

        // 4. 速度的插值
        std::vector<double2> velocity_qr(_eulerian_quadrature_rules.size());
        basic_mesh.interpolate_velocity(eulerian_velocity_u, eulerian_velocity_v, velocity_qr,
                                        _eulerian_quadrature_rules, make_double2(domain_mesh._dx, domain_mesh._dy),
                                        domain_mesh.get_dim_u(), domain_mesh.get_dim_v());

        auto dolfin_x = std::make_shared<dolfin::Function>(flow_past_cylinder->V);
        auto dolfin_b = std::make_shared<dolfin::Function>(flow_past_cylinder->V);
        auto dolfin_A = std::make_shared<dolfin::Matrix>(flow_past_cylinder->A);

        std::vector<double2> rhs(basic_mesh.num_dofs());
        basic_mesh.assemble_rhs_with_values_on_quadrature(velocity_qr, rhs);
        dolfin_b->vector()->set_local(algebra::flatten<double2, double>(rhs));

        dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");
        dolfin_x->vector()->get_local(U);
    }

    virtual void h(const std::vector<double>& X, std::vector<double>& Y) override {
        CHECK_F(X.size() == Y.size());
        std::vector<double> U(X.size());
        std::vector<double> X_ref = algebra::flatten<double2, double>(_solid_reference_configuration);
        std::vector<double> D(X.size());

        // 计算固体位移 D = X - X_ref
        algebra::axpy(-1, X_ref, X, D);

        calculate_solid_velocity(D, U);

        // 向后欧拉格式离散
        // (X - _solid_displacement_2) - dt*U = 0
        // Y = X - _solid_displacement_2
        // Y = Y - dt*U
        algebra::axpy(-1, _solid_displacement_2, D, Y);
        algebra::axpy(-dt, U, Y);
        LOG_F(INFO, "Objective function $\\|\\mathcal{X}\\|_2$ = %.16e", algebra::norm<double, double>(X));
        LOG_F(INFO, "Objective function $\\|\\mathcal{Y}\\|_2$ = %.16e", algebra::norm<double, double>(Y));
        LOG_F(INFO, "Objective function $\\|\\mathcal{X_ref}\\|_2$ = %.16e", algebra::norm<double, double>(X_ref));
    }

    double kinematic_energy(const std::vector<std::vector<double>>& eulerian_u,
                            const std::vector<std::vector<double>>& eulerian_v) {
        auto result = domain_mesh.cell_center_velocity(eulerian_u, eulerian_v);

        // \int_{\Omega}  1/2\rho |u|^2 d\Omega
        double ke = 0.5 * algebra::inner(result, result) * domain_mesh._rho * domain_mesh._dx * domain_mesh._dy;
        LOG_F(INFO, "动能 :  %.16e .", ke);

        return ke;
    }

    double eulerian_area() {
        std::vector<double2> vertices_displacement(basic_mesh.vertices.size());
        std::vector<double3> vertices_position(basic_mesh.vertices.size());

        basic_mesh.polynomial.evaluate_vertices<double2, double>(_solid_displacement.data(), basic_mesh.dofmaps.data(),
                                                                 basic_mesh.cells.data(), basic_mesh.vertices.data(),
                                                                 vertices_displacement.data(), basic_mesh.num_cells());

        // 使用 Lambda 函数对两个向量的对应元素进行相加
        std::transform(basic_mesh.vertices.begin(), basic_mesh.vertices.end(), vertices_displacement.begin(),
                       vertices_position.begin(),
                       [](double3 a, double2 b) { return make_double3(a.x + b.x, a.y + b.y, 0.0); });

        std::vector<double> areas(basic_mesh.num_cells());
        basic_mesh.polynomial.triangle_area_all_cells(basic_mesh.cells.data(), vertices_position.data(), areas.data(),
                                                      basic_mesh.num_cells());
        double areas_sum = std::accumulate(areas.begin(), areas.end(), 0.0);
        LOG_F(INFO, "固体单元数：%zu,当前固体面积 %.16e, 初始固体面积 %.16e", areas.size(), areas_sum,
              std::accumulate(basic_mesh.volumes.begin(), basic_mesh.volumes.end(), 0.0));
        return areas_sum;
    }
};

void calculate_position(std::vector<double4>& _eulerian_quadrature_rules, const std::vector<double4>& quadrature_rules,
                        const std::vector<double2>& displacement_qr) {
    CHECK_F(displacement_qr.size() == _eulerian_quadrature_rules.size());
    // 此处w分量不能重复相加
    for (size_t i = 0; i < displacement_qr.size(); i++) {
        _eulerian_quadrature_rules[i].x = quadrature_rules[i].x + displacement_qr[i].x;
        _eulerian_quadrature_rules[i].y = quadrature_rules[i].y + displacement_qr[i].y;
    }
}

void solve_one_step(ImmersedBoundaryMethod& ibm) {
    static int count = 0;
    count++;
    LOG_F(INFO, "调用 solve_one_step 次数： %d", count);

    // 计算高斯点处的固体位移
    std::vector<double2> displacement_qr(ibm._eulerian_quadrature_rules.size());
    ibm.basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
        ibm._solid_displacement.data(), ibm.basic_mesh.dofmaps.data(), displacement_qr.data(),
        ibm.basic_mesh.num_cells());

    calculate_position(ibm._eulerian_quadrature_rules, ibm.basic_mesh.quadrature_rules, displacement_qr);

    // 牛顿迭代的初始值
    // x = - ibm._solid_displacement_old_2 + 2*ibm._solid_displacement_2;
    std::vector<double> x = ibm._solid_displacement_old_2;
    algebra::scale(-1.0, x);
    auto temp_x = algebra::flatten<double2, double>(ibm._solid_reference_configuration);
    algebra::axpy(2.0, ibm._solid_displacement_2, x); // 位移
    algebra::axpy(1.0, temp_x, x);                    // 坐标

    // 牛顿迭代的参数
    size_t linear_max_iteration    = 10;   // bicgstab迭代的最大次数
    double linear_tol_i            = 1e-3; // 每次bicgstab迭代的残差
    double linear_tol              = 1e-3; // bicgstab迭代的总残差
    double nonlinear_tol           = 1e-4; // newton迭代的残差
    double nonlinear_max_iteration = 1;    // newton迭代的最大次数
    bool   silent                  = false;

    // 牛顿迭代
    newton_raphson(x, ibm, linear_max_iteration, linear_tol_i, linear_tol, nonlinear_tol, nonlinear_max_iteration,
                   silent);

    // 计算固体速度
    algebra::axpy(-1, temp_x, x);
    ibm.calculate_solid_velocity(x, ibm._solid_velocities_2);

    // 更新固体求解器中的速度
    ibm.flow_past_cylinder->Velocity->vector()->set_local(ibm._solid_velocities_2);

    auto [u, v, p, un, vn, uf, vf] = ibm.pipe_flow.get_variables();

    // 更新流体求解器中 un 和 vn
    un = algebra::ripple(ibm.eulerian_velocity_u, un.size());
    vn = algebra::ripple(ibm.eulerian_velocity_v, vn.size());

    // 位移的更新
    ibm._solid_velocities = algebra::ripple<double2, double>(ibm._solid_velocities_2);
    algebra::axpy(ibm.dt, ibm._solid_velocities, ibm._solid_displacement);
    ibm._solid_displacement_old_2 = ibm._solid_displacement_2;
    ibm._solid_displacement_2     = algebra::flatten<double2, double>(ibm._solid_displacement);

    // 存储固体力和固体位移
    if (count % 100 == 0)
        ibm.flow_past_cylinder->record<double2, double>(ibm._solid_forces, ibm._solid_displacement, ibm.t);
    ibm.eulerian_area();
    ibm.kinematic_energy(un, vn);
    // ibm.flow_past_cylinder->energy_norm(ibm._solid_displacement_2);

    // 存储流固耦合系统的速度场、体积力场、压强场
    LOG_F(INFO, "Recording velocity, pressure and force.");
    auto cell_center_velocity = ibm.domain_mesh.cell_center_velocity(un, vn);
    auto domain_mesh          = ibm.domain_mesh;
    if (count % 100 == 0)
        ibm.pipe_flow.velocity_writer.write(
            domain_mesh.get_dim(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
            make_double2(domain_mesh._dx, domain_mesh._dy), algebra::flatten(cell_center_velocity), ibm.t);
}

void test_1_2(int Nt, int N_s, int N_bg, double T, double kappa, double eta, double G_s, double nv, std::string path) {
    // 参数
    double Lx    = 2.46;
    double Ly    = 0.41;
    int    Nx    = 6 * N_bg;
    int    Ny    = N_bg;
    double mu_f  = 0.001;
    double rho_f = 1.0;
    double dt    = T / Nt;
    std::cout << dt << std::endl;

    // 读取固体网格
    auto mesh_file = geometry_path("temp/bar_behind_cylinder_");
    mesh_file += std::to_string(N_s);
    mesh_file += ".xdmf";
    std::cout << "Reading solid mesh.\n";
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    BasicMesh<DEGREE, DIM> basic_mesh(solid_mesh_dolfin);

    auto boundary_file = geometry_path("temp/bar_behind_cylinder_");
    boundary_file += std::to_string(N_s);
    boundary_file += "_domains.xml";
    auto materal_types = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, boundary_file);

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(Nt,                       // Nt
                                    {(size_t)Nx, (size_t)Ny}, // Nx Ny
                                    {Lx, Ly},                 // Lx Ly
                                    T,                        // T
                                    rho_f,                    // rho
                                    mu_f                      // mu
    );

    // 固体求解器
    auto flow_past_cylinder
        = std::make_shared<dolfin::BarBehindCylinder>(solid_mesh_dolfin, materal_types, kappa, eta, G_s, nv, path);

    // 流体求解器
    PipeFlow pipe_flow(domain_mesh, path + "fluid/velocity/data.pvd", path + "fluid/pressure/data.pvd",
                       path + "fluid/force/data.pvd");

    ImmersedBoundaryMethod test_flow_past_cylinder(basic_mesh, domain_mesh, pipe_flow, flow_past_cylinder);

    // 开始循环
    while (test_flow_past_cylinder.t < test_flow_past_cylinder.T - EPSILON) {
        test_flow_past_cylinder.t += test_flow_past_cylinder.dt;
        test_flow_past_cylinder.pipe_flow._t += test_flow_past_cylinder.pipe_flow._dt;
        solve_one_step(test_flow_past_cylinder);
    }
}

int main(int argc, char* argv[]) {
    int    N_s   = std::stoi(argv[1]);
    int    N_bg  = std::stoi(argv[2]);
    int    Nt    = std::stoi(argv[3]);
    double T     = std::stod(argv[4]);
    double kappa = std::stod(argv[5]);
    double eta   = std::stod(argv[6]);
    double G_s   = std::stod(argv[7]);
    double nv    = std::stod(argv[8]);

    std::ostringstream path;
    path << "demo_BarBehindCylinder_2D_implicit_" << N_s << "_" << N_bg << "_" << Nt << "_" << T << "_" << kappa << "_"
         << eta << "_" << G_s << "_" << nv << "/";

    loguru::add_file((path.str() + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file((path.str() + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    test_1_2(Nt, N_s, N_bg, T, kappa, eta, G_s, nv, path.str());
}

// make implicit_ibm_bar_behind_cylinder
// nohup ./test/implicit_ibm_bar_behind_cylinder 20 128 100000 5 1000000 0 2000
// 0.499 &

// demo_BarBehindCylinder_2D_implicit_  10    64     500000   100   1e+06   1000
// 10   100
//                                      N_s   N_bg   Nt       T     kappa   eta
//                                      mu   lamb

// ./test/implicit_ibm_bar_behind_cylinder 10 64 500000 100 1000000 1000 10 100