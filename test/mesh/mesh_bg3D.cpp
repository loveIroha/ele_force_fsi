/// @date 2023-10-08
/// @file mesh_bg3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///        1. 测试延拓算子
///        2. 测试插值算子

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh3D.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/DistributionInterpolation3D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven/DiskDriven.h>
#include <config.h>
#include <dolfin.h>
#include <io/ScopeProfiler.h>

#include <tuple>

const int DIM = 3;

double3 function_any_vector(const double3& x) {
    ScopeProfiler _{__func__};
    return make_double3(x.x, x.y, x.z);
}

double function_any_source(const double3& x) {
    ScopeProfiler _{__func__};
    return x.y;
}

void test_interpolation(std::shared_ptr<ImmersedMesh> solid_mesh, BackgroundMesh3D<3> domain_mesh) {
    ScopeProfiler _{__func__};

    // 创建定义在背景网格上的速度场
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh.set_function_on_staggered_grid(function_any_vector);

    // 创建定义在拉格朗日网格上的速度场
    [[maybe_unused]] std::vector<double3>& solid_velocities = solid_mesh->create_function<double3>("velocities");

    // 创建定义在拉格朗日网格积分点上的速度场
    std::vector<double3> velocities_for_quadrature;
    velocities_for_quadrature.resize(solid_mesh->get_quadrature_rules().size());
    auto quadrature_rules = solid_mesh->get_quadrature_rules();
    assert(quadrature_rules.size() == velocities_for_quadrature.size());

    // 拆分拉格朗日网格积分点上的速度场
    auto [lagrange_u, lagrange_v, lagrange_w] = algebra::split(velocities_for_quadrature);

    // 通过拉格朗日网格上的速度场插值得到背景网格上的速度场
    auto dim_u = domain_mesh.get_dim_u();
    auto dim_v = domain_mesh.get_dim_v();
    auto dim_w = domain_mesh.get_dim_w();

    interactor::interpolate_u(lagrange_u.data(), eulerian_u.data(), lagrange_u.size(), dim_u.x, dim_u.y, dim_u.z,
                              domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    interactor::interpolate_v(lagrange_v.data(), eulerian_v.data(), lagrange_v.size(), dim_v.x, dim_v.y, dim_v.z,
                              domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    interactor::interpolate_w(lagrange_w.data(), eulerian_w.data(), lagrange_w.size(), dim_w.x, dim_w.y, dim_w.z,
                              domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    velocities_for_quadrature = algebra::merge(lagrange_u, lagrange_v, lagrange_w);

    // // 输出速度在中心点上的值
    // for (size_t i = 0; i < velocities_for_quadrature.size(); i++)
    // {
    //     /* code */
    //     LOG_F(INFO, "quadrature points:         %.20f, %.20f, %.20f.",
    //     quadrature_rules[i].x, quadrature_rules[i].y, quadrature_rules[i].z);
    //     LOG_F(INFO, "velocities_for_quadrature: %.20f, %.20f, %.20f.",
    //     lagrange_u[i], lagrange_v[i], lagrange_w[i]);
    // }

    // 创建Dolfin变量
    dolfin::DiskDriven disk_driven(solid_mesh->get_dolfin_mesh());
    auto               dolfin_x = std::make_shared<dolfin::Function>(disk_driven.V);
    auto               dolfin_b = std::make_shared<dolfin::Function>(disk_driven.V);
    auto               dolfin_A = std::make_shared<dolfin::Matrix>(disk_driven.A);

    // 组装右端项
    LOG_F(INFO, "Interpolation operator.");
    std::vector<double3> rhs(solid_mesh->num_dofs());
    solid_mesh->assemble_rhs_with_values_on_quadrature(velocities_for_quadrature, rhs);
    dolfin_b->vector()->set_local(algebra::flatten<double3, double>(rhs));
    dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");

    // // 计算并打印误差
    // dolfin::Array<double> values(3);
    // dolfin::Array<double> x(3);
    // auto cooridnates = solid_mesh->get_dolfin_mesh()->coordinates();
    // for (size_t i = 0; i < cooridnates.size() / DIM; i++)
    // {
    //     x[0] = cooridnates[i * 3 + 0];
    //     x[1] = cooridnates[i * 3 + 1];
    //     x[2] = cooridnates[i * 3 + 2];
    //     LOG_F(INFO, "cooridnates: %.20f %.20f %.20f", cooridnates[i * 3 + 0],
    //     cooridnates[i * 3 + 1], cooridnates[i * 3 + 2]); dolfin_x->eval(values,
    //     x); auto exact = function_any_vector(make_double3(x[0], x[1], x[2]));
    //     LOG_F(INFO, "values     : %.20f %.20f %.20f", values[0], values[1],
    //     values[2]); LOG_F(INFO, "error      : %.20f %.20f %.20f", exact.x -
    //     values[0], exact.y - values[1], exact.z - values[2]);
    // }

    // 通过dolfin输出数据
    LOG_F(INFO, "Save the results.");
    dolfin::File file("a.pvd");
    file << *dolfin_x;
}

void test_distribution(std::shared_ptr<ImmersedMesh> solid_mesh, BackgroundMesh3D<3> domain_mesh) {
    // const char *scope_name;
    ScopeProfiler _{__func__};
    // TODO: create_function_on_quadrature_points
    std::vector<double3>  forces_for_quadrature;
    std::vector<double3>& solid_forces     = solid_mesh->create_function<double3>("forces");
    auto                  quadrature_rules = solid_mesh->get_quadrature_rules();

    forces_for_quadrature.resize(solid_mesh->get_quadrature_rules().size());
    solid_mesh->set_function("forces", function_any_vector);
    assert(quadrature_rules.size() == forces_for_quadrature.size());

    solid_mesh->evaluate_function_on_quadrature_points<double3, double>(solid_forces, forces_for_quadrature);

    auto [lagrange_u, lagrange_v, lagrange_w] = algebra::split(forces_for_quadrature);
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh.set_function_on_staggered_grid(function_any_vector);

    auto dim_u = domain_mesh.get_dim_u();
    auto dim_v = domain_mesh.get_dim_v();
    auto dim_w = domain_mesh.get_dim_w();

    algebra::zero(eulerian_u);
    algebra::zero(eulerian_v);
    algebra::zero(eulerian_w);

    interactor::distribute_u(lagrange_u.data(), eulerian_u.data(), lagrange_u.size(), dim_u.x, dim_u.y, dim_u.z,
                             domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    interactor::distribute_v(lagrange_v.data(), eulerian_v.data(), lagrange_v.size(), dim_v.x, dim_v.y, dim_v.z,
                             domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    interactor::distribute_w(lagrange_w.data(), eulerian_w.data(), lagrange_w.size(), dim_w.x, dim_w.y, dim_w.z,
                             domain_mesh._dx, domain_mesh._dy, domain_mesh._dz, quadrature_rules.data());

    double error1 = 0.0;
    double error2 = 0.0;
    double error4 = 0.0;
    int    error3 = 0;
    for (int i = 0; i < dim_u.x; i++) {
        for (int j = 1; j < dim_u.y - 1; j++) {
            for (int k = 1; k < dim_u.z - 1; k++) {
                double x     = (i)*domain_mesh._dx;
                double y     = (j - 0.5) * domain_mesh._dy;
                double z     = (k - 0.5) * domain_mesh._dz;
                auto   exact = function_any_vector(make_double3(x, y, z));
                if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5) < 0.1 * 0.1) {
                    error1 += (eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x);
                    error2 += std::abs(eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x);
                    error3++;
                    error4 = std::max(error4, std::abs(eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x));
                    LOG_F(INFO, "distribution error u: %.20f ",
                          eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x);
                    eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y]
                        = eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x;
                } else {
                    eulerian_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] = 0.0;
                }
            }
        }
    }

    // for (int i = 1; i < dim_v.x - 1; i++)
    // {
    //     for (int j = 0; j < dim_v.y; j++)
    //     {
    //         for (int k = 1; k < dim_v.z - 1; k++)
    //         {
    //             double x = (i - 0.5) * domain_mesh._dx;
    //             double y = (j)*domain_mesh._dy;
    //             double z = (k - 0.5) * domain_mesh._dz;
    //             auto exact = function_any_vector(make_double3(x, y, z));
    //             if (
    //                 (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) *
    //                 (z - 0.5) < 0.1 * 0.1)
    //             {
    //                 LOG_F(INFO, "distribution error v: %.20f ", eulerian_v[i +
    //                 j * dim_v.x + k * dim_v.x * dim_v.y] - exact.y);
    //             }
    //         }
    //     }
    // }

    // for (int i = 1; i < dim_w.x - 1; i++)
    // {
    //     for (int j = 1; j < dim_w.y - 1; j++)
    //     {
    //         for (int k = 0; k < dim_w.z; k++)
    //         {
    //             double x = (i - 0.5) * domain_mesh._dx;
    //             double y = (j - 0.5) * domain_mesh._dy;
    //             double z = (k)*domain_mesh._dz;
    //             auto exact = function_any_vector(make_double3(x, y, z));
    //             if (
    //                 (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) *
    //                 (z - 0.5) < 0.1 * 0.1)
    //             {
    //                 LOG_F(INFO, "distribution error w: %.20f ", eulerian_w[i +
    //                 j * dim_w.x + k * dim_w.x * dim_w.y] - exact.z);
    //             }
    //         }
    //     }
    // }

    LOG_F(INFO, "error2 u: %.20f", error2 / error3);
    LOG_F(INFO, "error1 u: %.20f", error1 / error3);
    LOG_F(INFO, "error4 u: %.20f", error4);
    LOG_F(INFO, "error3 u: %d", error3);
    // basic_mesh.distribute_force(eulerian_u,
    //                             eulerian_v,
    //                             function_qr,
    //                             basic_mesh.quadrature_rules,
    //                             make_double2(domain_mesh._dx, domain_mesh._dy),
    //                             domain_mesh.get_dim_u(),
    //                             domain_mesh.get_dim_v());

    // for (size_t i = 0; i < quadrature_rules.size(); i++)
    // {
    //     LOG_F(INFO, "quadrature points:     %f, %f, %f, %.20f.",
    //     quadrature_rules[i].x, quadrature_rules[i].y, quadrature_rules[i].z,
    //     quadrature_rules[i].w); LOG_F(INFO, "forces_for_quadrature: %f, %f,
    //     %f.", forces_for_quadrature[i].x, forces_for_quadrature[i].y,
    //     forces_for_quadrature[i].z);
    // }

    // 输出速度在中心点上的值
    auto        origin_u        = domain_mesh.get_origin_u();
    auto        dh              = domain_mesh.get_dh();
    std::string filename_center = "data/test_u.vti";
    write_vtk(dim_u, origin_u, dh, eulerian_u, filename_center, "u");

    auto origin_v   = domain_mesh.get_origin_v();
    filename_center = "data/test_v.vti";
    write_vtk(dim_v, origin_v, dh, eulerian_v, filename_center, "v");

    auto origin_w   = domain_mesh.get_origin_w();
    filename_center = "data/test_w.vti";
    write_vtk(dim_w, origin_w, dh, eulerian_w, filename_center, "w");
}

void test_interpolation_raw(BackgroundMesh3D<3> domain_mesh) {
    ScopeProfiler _{__func__};

    // 给eulerian_u和eulerian_v赋值
    auto dim_u                                = domain_mesh.get_dim_u();
    auto dim_v                                = domain_mesh.get_dim_v();
    auto dim_w                                = domain_mesh.get_dim_w();
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh.set_function_on_staggered_grid(function_any_vector);

    double  result[2];
    double4 qr[2];
    qr[0] = {0.3333333333333333333333, 0.22, 0.4, 0};
    qr[1] = {0.12, 0.23, 0.3, 0};

    interactor::interpolate_u(result, eulerian_u.data(), 2, dim_u.x, dim_u.y, dim_u.z, domain_mesh._dx, domain_mesh._dy,
                              domain_mesh._dz, qr);
    LOG_F(INFO, "u: %.20f %.20f", result[0], result[1]);

    interactor::interpolate_v(result, eulerian_v.data(), 2, dim_v.x, dim_v.y, dim_v.z, domain_mesh._dx, domain_mesh._dy,
                              domain_mesh._dz, qr);
    LOG_F(INFO, "v: %.20f %.20f", result[0], result[1]);

    interactor::interpolate_w(result, eulerian_w.data(), 2, dim_w.x, dim_w.y, dim_w.z, domain_mesh._dx, domain_mesh._dy,
                              domain_mesh._dz, qr);
    LOG_F(INFO, "w: %.20f %.20f", result[0], result[1]);
}

int main() {
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "插值算子");

    int3 dim = {32, 32, 32};
    // center at (0.6,0.5,0.5)
    dolfin::Point p2(0.2, 0.2, 0.2);
    dolfin::Point p3(0.8, 0.8, 0.8);
    auto          solid_mesh = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
        {p2, p3}, {(size_t)dim.x, (size_t)dim.y, (size_t)dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    BackgroundMesh3D<3> domain_mesh(10,              // Nt
                                    {100, 100, 100}, // Nx Ny
                                    {1.0, 1.0, 1.0}, // Lx Ly
                                    1.0,             // T
                                    1.0,             // rho
                                    0.01             // mu
    );

    for (size_t i = 0; i < 100; i++) {
        test_interpolation_raw(domain_mesh);
        test_interpolation(solid_mesh, domain_mesh);
        test_distribution(solid_mesh, domain_mesh);
    }

    printScopeProfiler();
}