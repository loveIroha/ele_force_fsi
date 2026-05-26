/// @date 2023-12-10
/// @file ImmersedBoundaryMethod3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#pragma once
#include <AlgebraSolver/algebra.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <io.h>
#include <vector_types.h>

#include <vector>

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

template <typename FluidSolver, typename SolidSolver>
class ImmersedBoundaryMethod : public algebra::NonlinearProblem {
  public:
    double _t  = 0.0;
    double _dt = 0.0;

    void set_dt(double dt) {
        _dt               = dt;
        domain_mesh->_dt  = _dt;
        solid_solver->_dt = _dt;
        fluid_solver->set_dt(_dt);
    }

    void set_t(double t) {
        _t               = t;
        domain_mesh->_t  = _t;
        solid_solver->_t = _t;
        fluid_solver->set_t(_t);
    }

    std::shared_ptr<BackgroundMesh3D<3>> domain_mesh;
    std::shared_ptr<FluidSolver>         fluid_solver;
    std::shared_ptr<ImmersedMeshP1>      solid_mesh;
    std::shared_ptr<SolidSolver>         solid_solver;

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
    ImmersedBoundaryMethod(std::shared_ptr<BackgroundMesh3D<3>> _domain_mesh,
                           std::shared_ptr<FluidSolver> _fluid_solver, std::shared_ptr<ImmersedMeshP1> _solid_mesh,
                           std::shared_ptr<SolidSolver> _solid_solver)
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
        // 设置时间 _dt
        set_dt(fluid_solver->_dt);
        set_t(0.0);
        solid_mesh->set_function("_solid_location_ref", function_location);
    }

    virtual void h(const std::vector<double>& x, std::vector<double>& grad) override {
        // 1. 将 x 转换成 std::vector<double3> 类型
        std::vector<double3> X(x.size() / 3);
        std::vector<double3> U(x.size() / 3);
        std::vector<double3> Y(x.size() / 3);
        LOG_F(INFO, "固体当前构型坐标的范数: %.16e", algebra::norm(x));

        for (size_t i = 0; i < X.size(); i++) {
            X[i].x = x[3 * i] - _solid_location_ref[i].x;
            X[i].y = x[3 * i + 1] - _solid_location_ref[i].y;
            X[i].z = x[3 * i + 2] - _solid_location_ref[i].z;
        }
        LOG_F(INFO, "固体位移的范数: %.16e", algebra::norm(X));

        // 计算固体位移 D = X - X_ref
        calculate_solid_velocity(X, U);

        LOG_F(INFO, "固体速度的范数: %.16e", algebra::norm(U));
        // 3. Y = X - _solid_displacement - dt * U;
        for (size_t i = 0; i < X.size(); i++) {
            Y[i].x = X[i].x - _solid_displacement[i].x - fluid_solver->_dt * U[i].x;
            Y[i].y = X[i].y - _solid_displacement[i].y - fluid_solver->_dt * U[i].y;
            Y[i].z = X[i].z - _solid_displacement[i].z - fluid_solver->_dt * U[i].z;
        }
        LOG_F(INFO, "残差 $\\|\\mathcal{Y}\\|_2$ : = %.16e", algebra::norm(Y));

        // 4. 将 Y 转换回 `Eigen::VectorXd` 类型
        for (size_t i = 0; i < X.size(); i++) {
            grad[3 * i]     = Y[i].x;
            grad[3 * i + 1] = Y[i].y;
            grad[3 * i + 2] = Y[i].z;
        }

        LOG_F(INFO, "位移的范数 $\\|\\mathcal{X}\\|_2$ = %.16e", algebra::norm(X));
    }

    void calculate_solid_velocity(const std::vector<double3>& X, std::vector<double3>& U) {
        // 统计计算耗时
        ScopeProfiler _{__func__};

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
        LOG_F(INFO, "Calculate the displacement of the solid at the Gauss point "
                    "(update the extension operator and the interpolation operator).");
        solid_mesh->polynomial->evaluate_quadrature_points<double3, double>(
            _solid_displacement.data(), solid_mesh->dofmaps.data(), displacement_qr.data(), solid_mesh->num_cells());
        calculate_position(quadrature_rules_cur, quadrature_rules_ref, displacement_qr);
    }
};

template <typename LocalFluidSolver, typename LocalSolidSolver>
void fsi_lid_simulation_explicit(std::shared_ptr<BackgroundMesh3D<3>> domain_mesh,
                                 std::shared_ptr<LocalFluidSolver>    fluid_solver,
                                 std::shared_ptr<ImmersedMeshP1>      solid_mesh,
                                 std::shared_ptr<LocalSolidSolver> solid_solver, int every_record = 50) {
    // 创建流固耦合对象
    ImmersedBoundaryMethod ibm{domain_mesh, fluid_solver, solid_mesh, solid_solver};

    int step = 0;
    while (ibm._t < ibm.fluid_solver->_Lt) {

        // if (ibm._t > 0.185) { ibm.set_dt(3.0e-5);} // 可以在程序运行过程中设置时间步长
        // ibm.set_dt(ibm._dt/10);  
        step++;
        ibm.set_t(ibm._t + ibm._dt); // 设置当前时间，同时设置流体求解器和固体求解器的当前时间
        printf("step : %d.\n", step);
        printf("current time of ibm          : %f, dt of ibm          : %f\n", ibm._t              , ibm._dt);
        printf("current time of fluid_solver : %f, dt of fluid_solver : %f\n", ibm.fluid_solver->_t, ibm.fluid_solver->_dt);
        printf("current time of solid_solver : %f, dt of solid_solver : %f\n", ibm.solid_solver->_t, ibm.solid_solver->_dt);
        printf("current time of domain_mesh  : %f, dt of domain_mesh  : %f\n", ibm.domain_mesh->_t , ibm.domain_mesh->_dt);
        // 计算高斯积分点处的固体位移（更新延拓算子和插值算子）
        ibm.fun_update_disp();

        // 计算固体速度
        ibm.calculate_solid_velocity(ibm._solid_displacement, ibm._solid_velocities);

        // 设置流体速度
        auto [un, vn, wn, pn]   = ibm.fluid_solver->get_velocity_and_pressure(); // 求解当前时刻速度场
        ibm.eulerian_velocity_u = algebra::flatten(un);                          // 获取当前时刻速度场
        ibm.eulerian_velocity_v = algebra::flatten(vn);
        ibm.eulerian_velocity_w = algebra::flatten(wn);

        // 计算固体位移
        LOG_F(INFO, "Calculate the displacement of the solid.");
        algebra::axpy(ibm._dt, ibm._solid_velocities, ibm._solid_displacement);

        // 输出结果
        if (step % every_record == 1) {
            LOG_F(INFO, "Record the results.");
            ibm.solid_solver->template record<double3, double>(ibm._solid_forces, ibm._solid_displacement, ibm._t);
            auto [f1, f2, f3]     = ibm.fluid_solver->get_source();
            auto [un, vn, wn, pn] = ibm.fluid_solver->get_velocity_and_pressure();
            ibm.fluid_solver->record(un, vn, wn, f1, f2, f3, pn, ibm._t);

            // TODO: 下面这些作为后处理函数，写入固体求解器中还是流体求解器中？
            auto kinematic_energy = ibm.fluid_solver->kinematic_energy(un, vn, wn);
            LOG_F(WATCH, "The kinematic_energy is %.16e.\n", kinematic_energy);

            auto potential_energy = ibm.solid_solver->energy_norm(ibm._solid_displacement_2);
            LOG_F(WATCH, "The potential_energy is %.16e.\n", potential_energy);

            // 输出固体当前构型的体积
            auto volumes_sum = ibm.solid_mesh->current_area(ibm._solid_displacement);
            LOG_F(WATCH, "The volume of the solid is %.16e.\n", volumes_sum);
        }
    }
}
