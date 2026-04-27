/// @date 2023-10-22
/// @file a.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven3D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow3D/LidDriven.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>

const int DIM = 3;

double3 function_any_vector(const double3& x) { return make_double3(x.x, x.y, x.z); }

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

int    Nt = 1000;
double T  = 10;

int                  Ns = 32;
int                  Nx = Ns;
int                  Ny = Ns;
int                  Nz = Ns;
std::array<int, DIM> dim_bg{Ns, Ns, Ns};

double                  Lx = 1.0;
double                  Ly = 1.0;
double                  Lz = 1.0;
std::array<double, DIM> L{Lx, Ly, Lz};

double rho  = 1.0;
double mu_f = 0.01;
double mu_s = 0.1;

int main() {
    ScopeProfiler _{__func__};
    // 创建背景网格
    auto domain_mesh = std::make_shared<BackgroundMesh3D<3>>(Nt, dim_bg, L, T, rho, mu_f);

    // 创建流体求解器
    auto fluid_solver = std::make_shared<stokes_flow::LidDriven<DIM>>(Nt, dim_bg, L, T, rho, mu_f);

    // 创建固体网格
    int3          dim = {16, 16, 16};
    dolfin::Point p2(0.4, 0.4, 0.4);
    dolfin::Point p3(0.6, 0.6, 0.6);
    auto          solid_mesh = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
        {p2, p3}, {(size_t)dim.x, (size_t)dim.y, (size_t)dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    // 创建固体求解器
    auto solid_solver = std::make_shared<dolfin::DiskDriven>(solid_mesh->get_dolfin_mesh(), mu_s);

    // 创建拉格朗日变量和欧拉变量的相互作用算子
    auto mesh_interator_3D = std::make_shared<MeshInteraction3D>(solid_mesh, domain_mesh);

    // 创建参考构型上的高斯积分法则
    auto quadrature_rules_ref = solid_mesh->get_quadrature_rules();

    // 将高斯积分点移动到当前构型上
    auto quadrature_rules_cur = solid_mesh->get_quadrature_rules();

    // 高斯积分点上的位移
    std::vector<double3> displacement_qr(quadrature_rules_ref.size());

    // 创建定义在拉格朗日网格上的物理场(dofs)
    std::vector<double3>& _solid_forces       = solid_mesh->create_function<double3>("forces");
    std::vector<double3>& _solid_velocities   = solid_mesh->create_function<double3>("velocities");
    std::vector<double3>& _solid_displacement = solid_mesh->create_function<double3>("displacement");

    auto _solid_forces_2       = algebra::flatten<double3, double>(_solid_forces);
    auto _solid_velocities_2   = algebra::flatten<double3, double>(_solid_velocities);
    auto _solid_displacement_2 = algebra::flatten<double3, double>(_solid_displacement);

    // 创建定义在背景网格上的速度场
    std::vector<double> eulerian_velocity_u(domain_mesh->get_size_u());
    std::vector<double> eulerian_velocity_v(domain_mesh->get_size_v());
    std::vector<double> eulerian_velocity_w(domain_mesh->get_size_w());

    // 创建定义在背景网格上的力场
    std::vector<double> eulerian_force_u(domain_mesh->get_size_u());
    std::vector<double> eulerian_force_v(domain_mesh->get_size_v());
    std::vector<double> eulerian_force_w(domain_mesh->get_size_w());

    for (int i = 1; i < Nt + 1; i++) {
        // 设置时间步长
        double dt        = T / Nt;
        double t         = i * dt;
        fluid_solver->_t = t;

        // 计算高斯积分点处的固体位移（更新延拓算子和插值算子）
        calculate_position(quadrature_rules_cur, quadrature_rules_ref, displacement_qr);

        // 计算固体力
        _solid_displacement_2 = algebra::flatten<double3, double>(_solid_displacement);
        solid_solver->solveOneStep(_solid_forces_2, _solid_displacement_2);
        _solid_forces = algebra::ripple<double3, double>(_solid_forces_2);

        // 调用分布算子
        mesh_interator_3D->distribute_force(eulerian_force_u, eulerian_force_v, eulerian_force_w, _solid_forces,
                                            quadrature_rules_cur);

        // 调用流体求解器
        fluid_solver->set_force(eulerian_force_u, eulerian_force_v,
                                eulerian_force_w); // 设置右端项
        // fluid_solver->set_velocity(eulerian_force_u, eulerian_force_v,
        // eulerian_force_w);    // 设置上一时刻速度场
        auto [uh, vh, wh, ph] = fluid_solver->solve(); // 求解当前时刻速度场
        eulerian_velocity_u   = algebra::flatten(uh);  // 获取当前时刻速度场
        eulerian_velocity_v   = algebra::flatten(vh);
        eulerian_velocity_w   = algebra::flatten(wh);

        // 调用插值算子
        mesh_interator_3D->interpolate_velocity(_solid_velocities, eulerian_velocity_u, eulerian_velocity_v,
                                                eulerian_velocity_w, quadrature_rules_cur);

        // 计算固体位移
        algebra::axpy(dt, _solid_velocities, _solid_displacement);
        solid_mesh->polynomial->evaluate_quadrature_points<double3, double>(
            _solid_displacement.data(), solid_mesh->dofmaps.data(), displacement_qr.data(), solid_mesh->num_cells());

        // 输出结果
        solid_solver->record<double3, double>(_solid_forces, _solid_displacement, t);
        fluid_solver->record(uh, vh, wh, ph, t); // TODO: 速度场和压力场的输出,还欠固体力的输出。
                                                 // fluid_solver->record(uh, vh, wh, ph, t); // TODO:
                                                 // 速度场和压力场的输出,还欠固体力的输出。
    }

    printScopeProfiler();
    return 0;
}