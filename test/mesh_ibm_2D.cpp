/// @date 2023-06-14
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试网格模块
///
///

#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven2D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <dolfin.h>
#include <io/writeVTK.h>

#include <tuple>

const int DEGREE = 1;
const int DIM    = 2;

// double2 function_any(const double3 &x)
// {
//     return make_double2(x.x, x.y);
// }

// double2 function_move(const double3 &x)
// {
//     return make_double2(-0.1, -0.2);
// }

// double2 function_displacement(const double3 &x)
// {
//     return make_double2(-0.1, 0.0);
// }

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

    LidDriven(BackgroundMesh2D<2> mesh)
        : stokes_flow{mesh._Nt, {mesh._Nx, mesh._Ny}, {mesh._Lx, mesh._Ly}, mesh._T, mesh._rho, mesh._mu} {
        // 设置边界类型
        int ubt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int vbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
        int pbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};

        double velocity_u[4] = {0, 1, 0, 0};
        double velocity_v[4] = {0, 0, 0, 0};
        double pressure[4]   = {0, 0, 0, 0};

        stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);

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

    auto get_variables() {
        return std::make_tuple(std::ref(u), std::ref(v), std::ref(p), std::ref(un), std::ref(vn), std::ref(uf),
                               std::ref(vf));
    }
};

class ImmersedBoundaryMethod {
  public:
    BasicMesh<DEGREE, DIM> basic_mesh;
    LidDriven              lid_driven;
    dolfin::DiskDriven     disk_driven;

    ImmersedBoundaryMethod(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<2> domain_mesh)
        : basic_mesh(solid_mesh_dolfin), lid_driven(domain_mesh), disk_driven(basic_mesh._mesh) {}

    void solve() {
        // 在固体网格上定义函数
        auto& _solid_forces              = basic_mesh.create_function<double2>("forces");
        auto& _solid_velocities          = basic_mesh.create_function<double2>("velocities");
        auto& _solid_displacement        = basic_mesh.create_function<double2>("displacement");
        auto  _solid_forces_2            = flatten<double2, double>(_solid_forces);
        auto  _solid_velocities_2        = flatten<double2, double>(_solid_velocities);
        auto  _solid_displacement_2      = flatten<double2, double>(_solid_displacement);
        auto  _eulerian_quadrature_rules = basic_mesh.quadrature_rules;

        // 在流体网格上定义函数
        std::vector<double> eulerian_force_u(domain_mesh.get_size_u());
        std::vector<double> eulerian_force_v(domain_mesh.get_size_v());
        std::vector<double> eulerian_velocity_u(domain_mesh.get_size_u());
        std::vector<double> eulerian_velocity_v(domain_mesh.get_size_v());
        auto&               _fluid_pressures = domain_mesh.create_function<double>("pressures");

        // 开始循环
        double t  = 0.0;
        double dt = domain_mesh._dt;
        while (t < domain_mesh._T - EPSILON) {
            t += dt;
            // 计算高斯点处的固体的位移
            std::vector<double2> displacement_qr(_eulerian_quadrature_rules.size());
            basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
                _solid_displacement.data(), basic_mesh.quadrature_rules, displacement_qr);

            // 计算高斯点处的固体的速度
            std::vector<double2> velocity_qr(_eulerian_quadrature_rules.size());
            basic_mesh.polynomial.evaluate_quadrature_points < double
        }
        int test_disk_driven(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<2> domain_mesh) {
            // 二维三角形网格、一阶多项式
            BasicMesh<DEGREE, DIM> basic_mesh(solid_mesh_dolfin);
            dolfin::DiskDriven     disk_driven(basic_mesh._mesh);
            LidDriven              lid_driven(domain_mesh);

            // 在固体网格上定义函数
            auto& _solid_forces              = basic_mesh.create_function<double2>("forces");
            auto& _solid_velocities          = basic_mesh.create_function<double2>("velocities");
            auto& _solid_displacement        = basic_mesh.create_function<double2>("displacement");
            auto  _solid_forces_2            = flatten<double2, double>(_solid_forces);
            auto  _solid_velocities_2        = flatten<double2, double>(_solid_velocities);
            auto  _solid_displacement_2      = flatten<double2, double>(_solid_displacement);
            auto  _eulerian_quadrature_rules = basic_mesh.quadrature_rules;

            // 在流体网格上定义函数
            std::vector<double>  eulerian_force_u(domain_mesh.get_size_u());
            std::vector<double>  eulerian_force_v(domain_mesh.get_size_v());
            std::vector<double>  eulerian_velocity_u(domain_mesh.get_size_u());
            std::vector<double>  eulerian_velocity_v(domain_mesh.get_size_v());
            std::vector<double>& _fluid_pressures = domain_mesh.create_function<double>("pressures");

            // 开始循环
            double t  = 0.0;
            double dt = domain_mesh._dt;
            while (t < domain_mesh._T - EPSILON) {
                t += dt;
                // 计算高斯点处的固体的位移
                std::vector<double2> displacement_qr(_eulerian_quadrature_rules.size());
                basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
                    _solid_displacement.data(), basic_mesh.dofmaps.data(), displacement_qr.data(),
                    basic_mesh.num_cells());

                for (size_t i = 0; i < displacement_qr.size(); i++) {
                    _eulerian_quadrature_rules[i].x = basic_mesh.quadrature_rules[i].x + displacement_qr[i].x;
                    _eulerian_quadrature_rules[i].y = basic_mesh.quadrature_rules[i].y + displacement_qr[i].y;
                }

                auto [u, v, p, un, vn, uf, vf] = lid_driven.get_variables();

                // 1. 求解固体的力
                disk_driven.solveOneStep(_solid_forces_2, _solid_displacement_2);
                _solid_forces = ripple<double2, double>(_solid_forces_2);

                // 2. 固体力的延拓
                std::vector<double2> force_qr(_eulerian_quadrature_rules.size());
                basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
                    _solid_forces.data(), basic_mesh.dofmaps.data(), force_qr.data(), basic_mesh.num_cells());
                basic_mesh.distribute_force(eulerian_force_u, eulerian_force_v, force_qr, _eulerian_quadrature_rules,
                                            make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_u(),
                                            domain_mesh.get_dim_v());
                // 输出延拓算子作用后的结果
                std::string filename_u  = "results/function_u.vti";
                std::string arrayname_u = "function_u";
                write_vtk(domain_mesh.get_dim_u(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
                          make_double2(domain_mesh._dx, domain_mesh._dy), eulerian_force_u, filename_u, arrayname_u);

                std::string filename_v  = "results/function_v.vti";
                std::string arrayname_v = "function_v";
                write_vtk(domain_mesh.get_dim_v(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
                          make_double2(domain_mesh._dx, domain_mesh._dy), eulerian_force_v, filename_v, arrayname_v);

                // 3. 求解速度场
                uf = ripple(eulerian_force_u, uf.size());
                vf = ripple(eulerian_force_v, vf.size());
                lid_driven.stokes_flow.solve_one_step(u, v, p, uf, vf, un, vn);
                un                  = u;
                vn                  = v;
                eulerian_velocity_u = flatten(u);
                eulerian_velocity_v = flatten(v);

                // 4. 速度的插值
                std::vector<double2> velocity_qr(_eulerian_quadrature_rules.size());
                basic_mesh.interpolate_velocity(
                    eulerian_velocity_u, eulerian_velocity_v, velocity_qr, _eulerian_quadrature_rules,
                    make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_u(), domain_mesh.get_dim_v());

                auto dolfin_x = std::make_shared<dolfin::Function>(disk_driven.V);
                auto dolfin_b = std::make_shared<dolfin::Function>(disk_driven.V);
                auto dolfin_A = std::make_shared<dolfin::Matrix>(disk_driven.A);

                std::vector<double2> rhs(basic_mesh.num_dofs());
                basic_mesh.assemble_rhs_with_values_on_quadrature(velocity_qr, rhs);
                dolfin_b->vector()->set_local(flatten<double2, double>(rhs));

                dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");
                dolfin::File("a.pvd") << *dolfin_x;
                dolfin_x->vector()->get_local(_solid_velocities_2);
                _solid_velocities = ripple<double2, double>(_solid_velocities_2);

                // auto data1 = _solid_velocities_2;
                // auto data2 =  _solid_velocities;
                // for (size_t logf_i = 0; logf_i < data1.size()/2; logf_i++)
                // {
                //     LOG_F(INFO, "gauss quadrature values: %d %.16e %.16e.", logf_i,
                //     data1[logf_i*2], data1[logf_i*2+1]); LOG_F(INFO, "gauss
                //     quadrature values: %d %.16e %.16e.", logf_i, data2[logf_i].x,
                //     data2[logf_i].y);
                // }

                // 5. 位移的更新
                double dt = domain_mesh._dt;
                for (size_t i = 0; i < _solid_displacement.size(); i++) {
                    _solid_displacement[i] = _solid_displacement[i] + dt * _solid_velocities[i];
                }
                _solid_displacement_2 = flatten<double2, double>(_solid_displacement);

                // auto data1 = _solid_displacement_2;
                // auto data2 =  _solid_displacement;
                // for (size_t logf_i = 0; logf_i < data1.size()/2; logf_i++)
                // {
                //     LOG_F(INFO, "gauss quadrature values: %d %.16e %.16e.", logf_i,
                //     data1[logf_i*2], data1[logf_i*2+1]); LOG_F(INFO, "gauss
                //     quadrature values: %d %.16e %.16e.", logf_i, data2[logf_i].x,
                //     data2[logf_i].y);
                // }

                // 存储固体力和固体位移
                disk_driven.record<double2, double>(_solid_forces, _solid_displacement, t);
            }

            return 0;
        }

        void test_1_2() {
            // 读取固体网格
            std::string mesh_file = "/public/home/fenics/Mesh_All_kinds_of_Flows/geometry/temp/circle_10.xdmf";
            std::cout << "Reading solid mesh.\n";
            dolfin::XDMFFile mesh_file_1(mesh_file);
            auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
            mesh_file_1.read(*solid_mesh_dolfin);
            mesh_file_1.close();
            std::cout << "Done.\n";
            std::cout << "Reading solid mesh.\n";

            // 背景网格
            BackgroundMesh2D<2> domain_mesh(8 * 128,    // Nt
                                            {128, 128}, // Nx Ny
                                            {1.0, 1.0}, // Lx Ly
                                            10.0,       // T
                                            1.0,        // rho
                                            0.01        // mu
            );

            test_disk_driven(solid_mesh_dolfin, domain_mesh);
        }

        int main() {
            loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
            loguru::add_file("warning.log", loguru::Append, loguru::Verbosity_WARNING);
            loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

            test_1_2();
        }
