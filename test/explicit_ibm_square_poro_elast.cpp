/// @date 2023-06-14
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试网格模块
///
///

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <PhysicsSolver/SolidSolver/SquarePoroElast2D/poroelasticmodel.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <config.h>
#include <dolfin.h>
#include <io/writeVTK.h>

#include <tuple>

const int   DEGREE = 1;
const int   DIM    = 2;


class LidDriven {
  public:
    StokesFlow<DIM>                  stokes_flow;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> v;
    std::vector<std::vector<double>> p;
    std::vector<std::vector<double>> un;
    std::vector<std::vector<double>> vn;
    std::vector<std::vector<double>> uf;
    std::vector<std::vector<double>> vf;
    std::vector<std::vector<double>> s;

    VTIWriter velocity_writer;
    VTIWriter pressure_writer;
    VTIWriter force_writer;

    LidDriven(BackgroundMesh2D<2> mesh, std::string velocity_file = "fluid/velocity/data.pvd",
              std::string pressure_file = "fluid/pressure/data.pvd", std::string force_file = "fluid/force/data.pvd")
        : stokes_flow{mesh._Nt, {mesh._Nx, mesh._Ny}, {mesh._Lx, mesh._Ly}, mesh._T, mesh._rho, mesh._mu},
          velocity_writer{velocity_file}, pressure_writer{pressure_file}, force_writer{force_file}

    {
        // 设置边界类型
        int ubt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};
        int vbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};
        int pbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};

        double velocity_u[4] = {0, 0, 0, 0};
        double velocity_v[4] = {0, 0, 0, 0};
        double pressure[4]   = {0, 0, 0, 0};

        stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);

        int _Nx = stokes_flow._Nx;
        int _Ny = stokes_flow._Ny;

        u = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        v = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        p = make_vector_2D<double>(_Nx + 2, _Ny + 2);
        s = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        un = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vn = make_vector_2D<double>(_Nx + 2, _Ny + 1);

        uf = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vf = make_vector_2D<double>(_Nx + 2, _Ny + 1);
    }

    auto get_variables() {
        return std::make_tuple(std::ref(u), std::ref(v), std::ref(p), std::ref(un), std::ref(vn), std::ref(uf),
                               std::ref(vf), std::ref(s));
    }
};

class ImmersedBoundaryMethod : public algebra::NonlinearProblem {
  public:
    BasicMesh<DEGREE, DIM>                          basic_mesh;
    LidDriven                                       lid_driven;
    std::shared_ptr<dolfin::PoroelasticModelSolver> poroelasticmodelsolver;
    BackgroundMesh2D<2>                             domain_mesh;

    // 在固体网格上定义函数
    // NOTE: 的变量会在calculate_velocity函数中被修改
    // NOTE: 非 TEMPORARY 的变量每个时间变量只会被修改一次
    std::vector<double2>& _solid_forces; // TEMPORARY
    std::vector<double2>& _solid_velocities;
    std::vector<double2>& _solid_displacement;
    std::vector<double>   _solid_forces_2; // TEMPORARY
    std::vector<double>   _solid_velocities_2;
    std::vector<double>   _solid_displacement_2;
    std::vector<double4>  _eulerian_quadrature_rules;
    std::vector<double>&  _solid_perfusion;

    // 在流体网格上定义函数
    std::vector<double>  eulerian_force_u;    // TEMPORARY
    std::vector<double>  eulerian_force_v;    // TEMPORARY
    std::vector<double>  eulerian_velocity_u; // TEMPORARY
    std::vector<double>  eulerian_velocity_v; // TEMPORARY
    std::vector<double>& _fluid_pressures;
    std::vector<double>  _fluid_perfusion;

    double  t;
    double& dt;
    double& T;

    ImmersedBoundaryMethod(BasicMesh<DEGREE, DIM> _solid_mesh, BackgroundMesh2D<DIM> _domain_mesh, LidDriven ns_solver,
                           std::shared_ptr<dolfin::PoroelasticModelSolver> solid_solver)
        : basic_mesh(_solid_mesh), lid_driven(ns_solver), poroelasticmodelsolver(solid_solver),
          domain_mesh(_domain_mesh), _solid_forces(basic_mesh.create_function<double2>("forces")),
          _solid_velocities(basic_mesh.create_function<double2>("velocities")),
          _solid_displacement(basic_mesh.create_function<double2>("displacement")),
          _solid_forces_2(algebra::flatten<double2, double>(_solid_forces)),
          _solid_velocities_2(algebra::flatten<double2, double>(_solid_velocities)),
          _solid_displacement_2(algebra::flatten<double2, double>(_solid_displacement)),
          _eulerian_quadrature_rules(basic_mesh.quadrature_rules), eulerian_force_u(domain_mesh.get_size_u()),
          eulerian_force_v(domain_mesh.get_size_v()), eulerian_velocity_u(domain_mesh.get_size_u()),
          eulerian_velocity_v(domain_mesh.get_size_v()),
          _fluid_pressures(domain_mesh.create_function<double>("pressures")), t(0.0), dt(lid_driven.stokes_flow._dt),
          T(lid_driven.stokes_flow._T), _fluid_perfusion(domain_mesh.get_size_p()),
          _solid_perfusion(basic_mesh.create_function<double>("perfusion"))

    {
        xk.resize(basic_mesh.num_dofs() * DIM);
        rk.resize(basic_mesh.num_dofs() * DIM);
    }

    void calculate_solid_velocity(const std::vector<double>& X, std::vector<double>& U) {
        static int count = 0;
        count++;
        LOG_F(INFO, "调用 calculate_solid_velocity 次数： %d", count);

        auto [u, v, p, un, vn, uf, vf, s] = lid_driven.get_variables();

        // 1.1 求解灌注源项
        poroelasticmodelsolver->solve_s(X);
        poroelasticmodelsolver->get_perfusion_source(_solid_perfusion);
        std::vector<double> _solid_perfusion_qr(_eulerian_quadrature_rules.size());
        basic_mesh.polynomial.evaluate_quadrature_points<double, double>(
            _solid_perfusion.data(), basic_mesh.dofmaps.data(), _solid_perfusion_qr.data(), basic_mesh.num_cells());

        basic_mesh.distribute_source(_fluid_perfusion, _solid_perfusion_qr, basic_mesh.quadrature_rules,
                                     make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_p());

        // std::string filename_s = "results/function_s.vti";
        // std::string arrayname_s = "function_s";
        // write_vtk(
        //     domain_mesh.get_dim_p(),
        //     make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
        //     make_double2(domain_mesh._dx, domain_mesh._dy),
        //     _fluid_perfusion,
        //     filename_s, arrayname_s);
        // 1. 求解固体的力
        // 修改了 _solid_forces_2 和 _solid_forces 变量
        std::vector<double> _solid_forces_2(X.size());
        poroelasticmodelsolver->solveOneStep(_solid_forces_2, X);
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
        // 修改了 lid_driven 中的 uf, vf
        // 修改了 eulerian_velocity_u, eulerian_velocity_v 两个变量
        uf = algebra::ripple(eulerian_force_u, uf.size());
        vf = algebra::ripple(eulerian_force_v, vf.size());
        s  = algebra::ripple(_fluid_perfusion, s.size());
        lid_driven.stokes_flow.solve_one_step(u, v, p, uf, vf, un, vn, s);

        eulerian_velocity_u = algebra::flatten(u);
        eulerian_velocity_v = algebra::flatten(v);

        // 4. 速度的插值
        std::vector<double2> velocity_qr(_eulerian_quadrature_rules.size());
        basic_mesh.interpolate_velocity(eulerian_velocity_u, eulerian_velocity_v, velocity_qr,
                                        _eulerian_quadrature_rules, make_double2(domain_mesh._dx, domain_mesh._dy),
                                        domain_mesh.get_dim_u(), domain_mesh.get_dim_v());

        auto dolfin_x = std::make_shared<dolfin::Function>(poroelasticmodelsolver->V);
        auto dolfin_b = std::make_shared<dolfin::Function>(poroelasticmodelsolver->V);
        auto dolfin_A = std::make_shared<dolfin::Matrix>(poroelasticmodelsolver->A);

        std::vector<double2> rhs(basic_mesh.num_dofs());
        basic_mesh.assemble_rhs_with_values_on_quadrature(velocity_qr, rhs);
        dolfin_b->vector()->set_local(algebra::flatten<double2, double>(rhs));

        dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");
        dolfin_x->vector()->get_local(U);
    }

    virtual void h(const std::vector<double>& X, std::vector<double>& Y) override {
        CHECK_F(X.size() == Y.size());
        std::vector<double> U(X.size());
        calculate_solid_velocity(X, U);

        // 向后欧拉格式离散
        // (X - _solid_displacement_2) - dt*U = 0
        // Y = X - _solid_displacement_2
        // Y = Y - dt*U
        algebra::axpy(-1, _solid_displacement_2, X, Y);
        algebra::axpy(-dt, U, Y);
        LOG_F(INFO, "Objective function $\\|\\mathcal{X}\\|_2$ = %.16e", algebra::norm<double, double>(X));
        LOG_F(INFO, "Objective function $\\|\\mathcal{Y}\\|_2$ = %.16e", algebra::norm<double, double>(Y));
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
    // 计算高斯点处的固体的位移
    std::vector<double2> displacement_qr(ibm._eulerian_quadrature_rules.size());
    ibm.basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
        ibm._solid_displacement.data(), ibm.basic_mesh.dofmaps.data(), displacement_qr.data(),
        ibm.basic_mesh.num_cells());

    calculate_position(ibm._eulerian_quadrature_rules, ibm.basic_mesh.quadrature_rules, displacement_qr);

    printf("ibm._solid_displacement size : %d\n", ibm._solid_displacement_2.size());
    ibm.calculate_solid_velocity(ibm._solid_displacement_2, ibm._solid_velocities_2);
    auto [u, v, p, un, vn, uf, vf, s] = ibm.lid_driven.get_variables();

    // 更新流体求解器中 un 和 vn
    un = algebra::ripple(ibm.eulerian_velocity_u, un.size());
    vn = algebra::ripple(ibm.eulerian_velocity_v, vn.size());

    // 位移的更新
    ibm._solid_velocities = algebra::ripple<double2, double>(ibm._solid_velocities_2);
    algebra::axpy(ibm.dt, ibm._solid_velocities, ibm._solid_displacement);
    ibm._solid_displacement_2 = algebra::flatten<double2, double>(ibm._solid_displacement);

    if (count % 100 == 1) {
        // 存储固体力和固体位移
        ibm.poroelasticmodelsolver->record<double2, double>(ibm._solid_forces, ibm._solid_displacement, ibm.t);
        ibm.poroelasticmodelsolver->record_1();

        // 存储流固耦合系统的速度场、体积力场、压强场
        LOG_F(INFO, "Recording velocity, pressure and force.");
        auto cell_center_velocity = ibm.domain_mesh.cell_center_velocity(un, vn);
        auto domain_mesh          = ibm.domain_mesh;
        ibm.lid_driven.velocity_writer.write(
            domain_mesh.get_dim(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
            make_double2(domain_mesh._dx, domain_mesh._dy), algebra::flatten(cell_center_velocity), ibm.t);
    }
}

void test_1_2(int Nt, int N_s, int N_bg, double T) {
    // 算例的目标文件夹
    std::ostringstream path;
    path << "demo_SquarePoroelast_2D_explicit_" << N_s << "_" << N_bg << "_" << Nt << "_" << T << "/";

    // 读取固体网格
    auto mesh_file = geometry_path("temp/square_poro_elast_");
    mesh_file += std::to_string(N_s);
    mesh_file += ".xdmf";
    std::cout << "Reading solid mesh.\n";
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    BasicMesh<DEGREE, DIM> basic_mesh(solid_mesh_dolfin);

    auto boundary_file = geometry_path("temp/square_poro_elast_");
    boundary_file += std::to_string(N_s);
    boundary_file += "_boundary.xml";
    auto boundaries = std::make_shared<dolfin::MeshFunction<std::size_t>>(solid_mesh_dolfin, boundary_file);

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(Nt,                           // Nt
                                    {(size_t)N_bg, (size_t)N_bg}, // Nx Ny
                                    {2.0, 2.0},                   // Lx Ly
                                    T,                            // T
                                    1.0,                          // rho
                                    0.04                          // mu
    );

    // 固体求解器
    auto poroelasticmodelsolver
        = std::make_shared<dolfin::PoroelasticModelSolver>(solid_mesh_dolfin, boundaries, T / Nt);

    // 流体求解器
    LidDriven lid_driven(domain_mesh, path.str() + "fluid/velocity/data.pvd", path.str() + "fluid/pressure/data.pvd",
                         path.str() + "fluid/force/data.pvd");

    ImmersedBoundaryMethod test_disk_driven(basic_mesh, domain_mesh, lid_driven, poroelasticmodelsolver);

    // 开始循环
    while (test_disk_driven.t < test_disk_driven.T - EPSILON) {
        test_disk_driven.t += test_disk_driven.dt;
        poroelasticmodelsolver->_t = test_disk_driven.t;
        solve_one_step(test_disk_driven);
    }
}

int main(int argc, char* argv[]) {
    int    N_s  = std::stoi(argv[1]);
    int    N_bg = std::stoi(argv[2]);
    int    Nt   = std::stoi(argv[3]);
    double T    = std::stod(argv[4]);

    std::ostringstream path;
    path << "demo_SquarePoroelast_2D_explicit_" << N_s << "_" << N_bg << "_" << Nt << "_" << T << "/";

    // loguru::add_file((path.str() + "everything.log").c_str(), loguru::Append,
    // loguru::Verbosity_MAX); loguru::add_file((path.str() +
    // "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
    // loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    test_1_2(Nt, N_s, N_bg, T);
}
