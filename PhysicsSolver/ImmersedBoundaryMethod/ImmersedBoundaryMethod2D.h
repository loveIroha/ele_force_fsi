/// @date 2023-06-19
/// @file ImmersedBoundaryMethod2D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <config.h>
#include <dolfin.h>

#include <algorithm>

double2 function_reference_configuration(const double3& x) { return make_double2(x.x, x.y); }
template <typename FluidSolver, typename SolidSolver>
class ImmersedBoundaryMethod : public algebra::NonlinearProblem {
  public:
    BasicMesh<DEGREE, DIM>       basic_mesh;
    FluidSolver                  _ns_solver;
    std::shared_ptr<SolidSolver> _solid_solver;
    BackgroundMesh2D<2>          domain_mesh;

    // 在固体网格上定义函数
    // NOTE: 的变量会在calculate_velocity函数中被修改
    // NOTE: 非 TEMPORARY 的变量每个时间变量只会被修改一次
    std::vector<double2>& _solid_forces;                  // TEMPORARY
    std::vector<double2>& _solid_velocities;              //
    std::vector<double2>& _solid_displacement;            //
    std::vector<double2>& _solid_displacement_old;        //
    std::vector<double2>& _solid_reference_configuration; //
    std::vector<double>   _solid_forces_2;                // TEMPORARY
    std::vector<double>   _solid_velocities_2;            //
    std::vector<double>   _solid_displacement_2;          //
    std::vector<double>   _solid_displacement_old_2;      //
    std::vector<double4>  _eulerian_quadrature_rules;     //

    // 在流体网格上定义函数
    std::vector<double>  eulerian_force_u;    // TEMPORARY
    std::vector<double>  eulerian_force_v;    // TEMPORARY
    std::vector<double>  eulerian_velocity_u; // TEMPORARY
    std::vector<double>  eulerian_velocity_v; // TEMPORARY
    std::vector<double>& _fluid_pressures;    //

    double  t;
    double& dt;
    double& T;

    ImmersedBoundaryMethod(BasicMesh<DEGREE, DIM> _solid_mesh, BackgroundMesh2D<DIM> _domain_mesh,
                           FluidSolver ns_solver, std::shared_ptr<SolidSolver> solid_solver)
        : basic_mesh(_solid_mesh), _ns_solver(ns_solver), _solid_solver(solid_solver), domain_mesh(_domain_mesh),
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
          _fluid_pressures(domain_mesh.create_function<double>("pressures")), t(0.0), dt(_ns_solver.stokes_flow._dt),
          T(_ns_solver.stokes_flow._T) {
        basic_mesh.set_function("reference_configuration", function_reference_configuration);
        xk.resize(basic_mesh.num_dofs() * DIM);
        rk.resize(basic_mesh.num_dofs() * DIM);
    }

    void calculate_solid_velocity(const std::vector<double>& X, std::vector<double>& U) {
        static int count = 0;
        count++;
        LOG_F(INFO, "调用 calculate_solid_velocity 次数： %d", count);

        auto [u, v, p, un, vn, uf, vf] = _ns_solver.get_variables();

        // 1. 求解固体的力
        // 修改了 _solid_forces_2 和 _solid_forces 变量
        std::vector<double> _solid_forces_2(X.size());
        _solid_solver->solveOneStep(_solid_forces_2, X);
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
        // 修改了 _ns_solver 中的 uf, vf
        // 修改了 eulerian_velocity_u, eulerian_velocity_v 两个变量
        uf = algebra::ripple(eulerian_force_u, uf.size());
        vf = algebra::ripple(eulerian_force_v, vf.size());
        _ns_solver.stokes_flow.solve_one_step(u, v, p, uf, vf, un, vn);
        eulerian_velocity_u = algebra::flatten(u);
        eulerian_velocity_v = algebra::flatten(v);

        // 4. 速度的插值
        std::vector<double2> velocity_qr(_eulerian_quadrature_rules.size());
        basic_mesh.interpolate_velocity(eulerian_velocity_u, eulerian_velocity_v, velocity_qr,
                                        _eulerian_quadrature_rules, make_double2(domain_mesh._dx, domain_mesh._dy),
                                        domain_mesh.get_dim_u(), domain_mesh.get_dim_v());

        auto dolfin_x = std::make_shared<dolfin::Function>(_solid_solver->V);
        auto dolfin_b = std::make_shared<dolfin::Function>(_solid_solver->V);
        auto dolfin_A = std::make_shared<dolfin::Matrix>(_solid_solver->A);

        std::vector<double2> rhs(basic_mesh.num_dofs());
        basic_mesh.assemble_rhs_with_values_on_quadrature(velocity_qr, rhs);
        dolfin_b->vector()->set_local(algebra::flatten<double2, double>(rhs));

        dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");
        dolfin_x->vector()->get_local(U);
    }

    // NOTE: X is the position of solid, not the displacement of the solid.
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
        double areas_sum          = std::accumulate(areas.begin(), areas.end(), 0.0);
        double areas_sum_eulerian = std::accumulate(basic_mesh.volumes.begin(), basic_mesh.volumes.end(), 0.0);
        LOG_F(INFO, "固体单元数：%zu,当前固体面积 %.16e, 初始固体面积 %.16e", areas.size(), areas_sum,
              areas_sum_eulerian);
        printf("面积：%f\n", (areas_sum - areas_sum_eulerian) / areas_sum_eulerian);
        return areas_sum;
    }
};

template <typename FluidSolver, typename SolidSolver>
void solve_one_step(ImmersedBoundaryMethod<FluidSolver, SolidSolver>& ibm) {
    LOG_SCOPE_FUNCTION(INFO);
    static int count = 0;
    count++;
    LOG_F(INFO, "调用 solve_one_step 次数： %d", count);
    // 计算高斯点处的固体的位移
    std::vector<double2> displacement_qr(ibm._eulerian_quadrature_rules.size());
    ibm.basic_mesh.polynomial.template evaluate_quadrature_points<double2, double>(
        ibm._solid_displacement.data(), ibm.basic_mesh.dofmaps.data(), displacement_qr.data(),
        ibm.basic_mesh.num_cells());

    calculate_position(ibm._eulerian_quadrature_rules, ibm.basic_mesh.quadrature_rules, displacement_qr);

    ibm.calculate_solid_velocity(ibm._solid_displacement_2, ibm._solid_velocities_2);
    auto [u, v, p, un, vn, uf, vf] = ibm._ns_solver.get_variables();

    // 更新流体求解器中 un 和 vn
    un = algebra::ripple(ibm.eulerian_velocity_u, un.size());
    vn = algebra::ripple(ibm.eulerian_velocity_v, vn.size());

    // 位移的更新
    ibm._solid_velocities = algebra::ripple<double2, double>(ibm._solid_velocities_2);
    algebra::axpy(ibm.dt, ibm._solid_velocities, ibm._solid_displacement);
    ibm._solid_displacement_old_2 = ibm._solid_displacement_2;
    ibm._solid_displacement_2     = algebra::flatten<double2, double>(ibm._solid_displacement);

    // 只记录100个时刻的数据
    int num_records = ibm.domain_mesh._Nt / 100;
    num_records     = std::max(num_records, 1);
    if (count % num_records == 1) {
        LOG_F(INFO, "Recording velocity, pressure and force.");

        // 存储固体力和固体位移
        ibm._solid_solver->template record<double2, double>(ibm._solid_forces, ibm._solid_displacement, ibm.t);
        ibm.eulerian_area();
        ibm.kinematic_energy(un, vn);
        ibm._solid_solver->energy_norm(ibm._solid_displacement_2);

        // 存储流固耦合系统的速度场、体积力场、压强场
        auto cell_center_velocity = ibm.domain_mesh.cell_center_velocity(un, vn);
        auto cell_center_force    = ibm.domain_mesh.cell_center_velocity(uf, vf);
        auto domain_mesh          = ibm.domain_mesh;
        ibm._ns_solver.writer.write(domain_mesh.get_dim(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
                                    make_double2(domain_mesh._dx, domain_mesh._dy),
                                    algebra::flatten(cell_center_velocity), algebra::flatten(cell_center_force),
                                    algebra::flatten(p), ibm.t);
    }
}

template <typename FluidSolver, typename SolidSolver>
void solve_one_step_implicit(ImmersedBoundaryMethod<FluidSolver, SolidSolver>& ibm) {
    LOG_SCOPE_FUNCTION(INFO);
    static int count = 0;
    count++;
    LOG_F(INFO, "调用 solve_one_step 次数： %d", count);
    // 计算高斯点处的固体的位移
    std::vector<double2> displacement_qr(ibm._eulerian_quadrature_rules.size());
    ibm.basic_mesh.polynomial.template evaluate_quadrature_points<double2, double>(
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
    auto [u, v, p, un, vn, uf, vf] = ibm._ns_solver.get_variables();

    // 更新流体求解器中 un 和 vn
    un = algebra::ripple(ibm.eulerian_velocity_u, un.size());
    vn = algebra::ripple(ibm.eulerian_velocity_v, vn.size());

    // 位移的更新
    ibm._solid_velocities = algebra::ripple<double2, double>(ibm._solid_velocities_2);
    algebra::axpy(ibm.dt, ibm._solid_velocities, ibm._solid_displacement);
    ibm._solid_displacement_old_2 = ibm._solid_displacement_2;
    ibm._solid_displacement_2     = algebra::flatten<double2, double>(ibm._solid_displacement);
    if (count % 100 == 1) {
        LOG_F(INFO, "Recording velocity, pressure and force.");

        // 存储固体力和固体位移
        ibm._solid_solver->template record<double2, double>(ibm._solid_forces, ibm._solid_displacement, ibm.t);
        ibm.eulerian_area();
        ibm.kinematic_energy(un, vn);
        ibm._solid_solver->energy_norm(ibm._solid_displacement_2);

        // 存储流固耦合系统的速度场、体积力场、压强场
        auto cell_center_velocity = ibm.domain_mesh.cell_center_velocity(un, vn);
        auto cell_center_force    = ibm.domain_mesh.cell_center_velocity(uf, vf);
        auto domain_mesh          = ibm.domain_mesh;
        ibm._ns_solver.writer.write(domain_mesh.get_dim(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
                                    make_double2(domain_mesh._dx, domain_mesh._dy),
                                    algebra::flatten(cell_center_velocity), algebra::flatten(cell_center_force),
                                    algebra::flatten(p), ibm.t);
    }
}

void calculate_position(std::vector<double4>& _eulerian_quadrature_rules, const std::vector<double4>& quadrature_rules,
                        const std::vector<double2>& displacement_qr) {
    CHECK_F(displacement_qr.size() == _eulerian_quadrature_rules.size());
    // 此处w分量不能重复相加
    for (size_t i = 0; i < displacement_qr.size(); i++) {
        _eulerian_quadrature_rules[i].x = quadrature_rules[i].x + displacement_qr[i].x;
        _eulerian_quadrature_rules[i].y = quadrature_rules[i].y + displacement_qr[i].y;
    }
}
