/// @date 2023-12-10
/// @file mesh_bg3D_kokkos.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh3D.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/DistributionInterpolation3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <config.h>
#include <dolfin.h>
#include <io/ScopeProfiler.h>

#include <tuple>

double3 function_any_vector_legacy(const double3& x) {
    ScopeProfiler _{__func__};
    return make_double3(x.x, x.y, x.z);
}

void test_interpolation_raw(std::shared_ptr<BackgroundMesh3D<3>> domain_mesh) {
    LOG_SCOPE_F(INFO, "插值算子");
    ScopeProfiler _{__func__};

    // 给eulerian_u和eulerian_v赋值
    auto dim_u = domain_mesh->get_dim_u();
    auto dim_v = domain_mesh->get_dim_v();
    auto dim_w = domain_mesh->get_dim_w();

    auto function_any_vector = [](double3 x) { return make_double3(x.x, x.y, x.z); };
    auto function_any_vector_4
        = [&function_any_vector](double4 x) { return function_any_vector(make_double3(x.x, x.y, x.z)); };
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh->set_function_on_staggered_grid(function_any_vector);

    double  result[2];
    double4 qr[2];
    qr[0] = {0.3333333333333333333333, 0.22, 0.4, 0};
    qr[1] = {0.12, 0.23, 0.3, 0};

    // 用于检查计算结果的函数
    auto check_results = [function_any_vector_4](const double* result, const double4* qr, int components = 0) {
        for (int i = 0; i < 2; i++) {
            double3 _exact = function_any_vector_4(qr[i]);
            double* exact  = (double*)&_exact;
            LOG_F(INFO, "u: %.20f %.20f", result[i], exact[components]);
            CHECK_F(std::abs(result[i] - exact[components]) < NPUHEART_EPS, "Error too large.");
        }
    };

    interactor::interpolate_u(result, eulerian_u.data(), 2, dim_u.x, dim_u.y, dim_u.z, domain_mesh->_dx,
                              domain_mesh->_dy, domain_mesh->_dz, qr);
    check_results(result, qr, 0);

    interpolate_u(result, eulerian_u.data(), qr, {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_u, 2);
    check_results(result, qr, 0);

    interactor::interpolate_v(result, eulerian_v.data(), 2, dim_v.x, dim_v.y, dim_v.z, domain_mesh->_dx,
                              domain_mesh->_dy, domain_mesh->_dz, qr);
    check_results(result, qr, 1);

    interpolate_v(result, eulerian_v.data(), qr, {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_v, 2);
    check_results(result, qr, 1);

    interactor::interpolate_w(result, eulerian_w.data(), 2, dim_w.x, dim_w.y, dim_w.z, domain_mesh->_dx,
                              domain_mesh->_dy, domain_mesh->_dz, qr);
    check_results(result, qr, 2);

    interpolate_w(result, eulerian_w.data(), qr, {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_w, 2);
    check_results(result, qr, 2);
}

int test_interpolation(std::shared_ptr<ImmersedMesh> solid_mesh, std::shared_ptr<BackgroundMesh3D<3>> domain_mesh) {
    auto mesh_interator_3D = std::make_shared<MeshInteraction3D>(solid_mesh, domain_mesh);

    auto function_any_vector                  = [](double3 x) { return make_double3(x.x, x.y, x.z); };
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh->set_function_on_staggered_grid(function_any_vector);
    auto& _solid_velocities                   = solid_mesh->create_function<double3>("velocities");
    auto& quadrature_rules                    = solid_mesh->get_quadrature_rules();

    // 调用插值算子
    LOG_F(INFO, "Interpolate the velocity of the fluid to the Lagrangian grid.");
    mesh_interator_3D->interpolate_velocity(_solid_velocities, eulerian_u, eulerian_v, eulerian_w, quadrature_rules);
    IO::write_vector_1D(_solid_velocities, "solid_velocities.txt");

    // 计算并打印误差
    dolfin::DiskDriven    disk_driven(solid_mesh->get_dolfin_mesh(), 0.1);
    auto                  dolfin_x        = std::make_shared<dolfin::Function>(disk_driven.V);
    auto                  dolfin_x_vector = algebra::flatten<double3, double>(_solid_velocities);
    auto                  cooridnates     = solid_mesh->get_dolfin_mesh()->coordinates();
    dolfin::Array<double> x(3);
    dolfin::Array<double> values(3);
    dolfin_x->vector()->set_local(dolfin_x_vector);
    auto max_error = 0.0;
    for (size_t i = 0; i < cooridnates.size() / DIM; i++) {
        x[0] = cooridnates[i * 3 + 0];
        x[1] = cooridnates[i * 3 + 1];
        x[2] = cooridnates[i * 3 + 2];
        LOG_F(INFO, "cooridnates: %.20f %.20f %.20f", cooridnates[i * 3 + 0], cooridnates[i * 3 + 1],
              cooridnates[i * 3 + 2]);
        dolfin_x->eval(values, x);
        auto exact = function_any_vector(make_double3(x[0], x[1], x[2]));
        LOG_F(INFO, "values     : %.20f %.20f %.20f", values[0], values[1], values[2]);
        LOG_F(INFO, "error      : %.20f %.20f %.20f", exact.x - values[0], exact.y - values[1], exact.z - values[2]);
        max_error = std::max(max_error, exact.x - values[0]);
        max_error = std::max(max_error, exact.y - values[1]);
        max_error = std::max(max_error, exact.z - values[2]);
        LOG_F(INFO, "max_error  : %.20f", max_error);
        CHECK_F(max_error < 0.00004, "Error too large.");
    }
    return 0;
}

int test_distribution(std::shared_ptr<ImmersedMesh> solid_mesh, std::shared_ptr<BackgroundMesh3D<3>> domain_mesh) {
    auto                  mesh_interator_3D = std::make_shared<MeshInteraction3D>(solid_mesh, domain_mesh);
    std::vector<double3>& solid_forces      = solid_mesh->create_function<double3>("forces");
    solid_mesh->set_function("forces", function_any_vector_legacy);
    auto function_any_vector = [](double3 x) { return make_double3(x.x, x.y, x.z); };

    // 创建定义在背景网格上的力场
    std::vector<double> eulerian_force_u(domain_mesh->get_size_u());
    std::vector<double> eulerian_force_v(domain_mesh->get_size_v());
    std::vector<double> eulerian_force_w(domain_mesh->get_size_w());
    auto&               quadrature_rules = solid_mesh->get_quadrature_rules();

    // 调用分布算子
    LOG_F(INFO, "Distribute the force of the solid to the Eulerian grid.");
    mesh_interator_3D->distribute_force(eulerian_force_u, eulerian_force_v, eulerian_force_w, solid_forces,
                                        quadrature_rules);

    // 背景网格的大小
    auto   dim_u     = domain_mesh->get_dim_u();
    auto   dim_v     = domain_mesh->get_dim_v();
    auto   dim_w     = domain_mesh->get_dim_w();
    double max_error = 0.0;
    // 计算结果
    // 1. 检查 eulerian_force_u
    for (int i = 1; i < dim_u.x - 1; i++) {
        for (int j = 1; j < dim_u.y - 1; j++) {
            for (int k = 1; k < dim_u.z - 1; k++) {
                double x      = (i - 1.0) * domain_mesh->_dx;
                double y      = (j - 0.5) * domain_mesh->_dy;
                double z      = (k - 0.5) * domain_mesh->_dz;
                auto   exact  = function_any_vector(make_double3(x, y, z));
                auto   radial = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);
                // 0.3 为边长的立方体; 0.2为半径的球体;
                if (radial < 0.1 * 0.1) {
                    auto local_error = (eulerian_force_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] - exact.x);
                    max_error        = std::max(max_error, std::abs(local_error));
                    LOG_F(INFO, "distribution error u: %.20f %.20f", local_error, max_error);
                }
            }
        }
    }
    for (int i = 1; i < dim_v.x - 1; i++) {
        for (int j = 1; j < dim_v.y - 1; j++) {
            for (int k = 1; k < dim_v.z - 1; k++) {
                double x      = (i - 0.5) * domain_mesh->_dx;
                double y      = (j - 1.0) * domain_mesh->_dy;
                double z      = (k - 0.5) * domain_mesh->_dz;
                auto   exact  = function_any_vector(make_double3(x, y, z));
                auto   radial = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);
                // 0.3 为边长的立方体; 0.2为半径的球体;
                if (radial < 0.1 * 0.1) {
                    auto local_error = (eulerian_force_v[i + j * dim_v.x + k * dim_v.x * dim_v.y] - exact.y);
                    max_error        = std::max(max_error, std::abs(local_error));
                    LOG_F(INFO, "distribution error v: %.20f %.20f", local_error, max_error);
                }
            }
        }
    }
    for (int i = 1; i < dim_w.x - 1; i++) {
        for (int j = 1; j < dim_w.y - 1; j++) {
            for (int k = 1; k < dim_w.z - 1; k++) {
                double x      = (i - 0.5) * domain_mesh->_dx;
                double y      = (j - 0.5) * domain_mesh->_dy;
                double z      = (k - 1.0) * domain_mesh->_dz;
                auto   exact  = function_any_vector(make_double3(x, y, z));
                auto   radial = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5);
                // 0.3 为边长的立方体; 0.2为半径的球体;
                if (radial < 0.1 * 0.1) {
                    auto local_error = (eulerian_force_w[i + j * dim_w.x + k * dim_w.x * dim_w.y] - exact.z);
                    max_error        = std::max(max_error, std::abs(local_error));
                    LOG_F(INFO, "distribution error w: %.20f %.20f", local_error, max_error);
                }
            }
        }
    }
    CHECK_F(max_error < 0.0003, "Error too large.");

    // 输出结果
    std::string filename;
    filename = "eulerian_force_u_kokkos.txt";
    IO::write_vector_3D(algebra::ripple(eulerian_force_u, dim_u.x, dim_u.y), filename);
    filename = "eulerian_force_v_kokkos.txt";
    IO::write_vector_3D(algebra::ripple(eulerian_force_v, dim_v.x, dim_v.y), filename);
    filename = "eulerian_force_w_kokkos.txt";
    IO::write_vector_3D(algebra::ripple(eulerian_force_w, dim_w.x, dim_w.y), filename);
    return 0;
}

int main(int argc, char* argv[]) {
    LOG_F(INFO, "创建背景网格");
    const std::array<int, DIM> dim = {64, 64, 64};
    dolfin::Point              p2(0.2, 0.2, 0.2);
    dolfin::Point              p3(0.8, 0.8, 0.8);
    auto solid_mesh        = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
        {p2, p3}, {(size_t)dim[0], (size_t)dim[1], (size_t)dim[2]}, dolfin::CellType::Type::tetrahedron)));
    auto solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    LOG_F(INFO, "创建固体网格");
    const std::array<double, DIM> L           = {1.0, 1.0, 1.0};
    const std::array<int, DIM>    dim_bg      = {64, 64, 64};
    auto                          domain_mesh = std::make_shared<BackgroundMesh3D<3>>(10,     // Nt
                                                             dim_bg, // Nx Ny Nz
                                                             L,      // Lx Ly Lz
                                                             1.0,    // T
                                                             1.0,    // rho
                                                             0.01    // mu
    );

    Kokkos::initialize(argc, argv);
    LOG_F(INFO, "测试%zu次", 1);
    for (size_t i = 0; i < 1; i++) {
        // test_interpolation_raw(domain_mesh);
        test_interpolation(solid_mesh, domain_mesh);
        // test_distribution(solid_mesh, domain_mesh);
    }
    Kokkos::finalize();
    // {
    //     auto a = IO::read_vector_3D<double>("eulerian_force_u.txt");
    //     auto b = IO::read_vector_3D<double>("eulerian_force_u_kokkos.txt");
    //     // 遍历三维向量
    //     for (size_t i = 0; i < a.size(); ++i)
    //     {
    //         for (size_t j = 0; j < a[i].size(); ++j)
    //         {
    //             for (size_t k = 0; k < a[i][j].size(); ++k)
    //             {
    //                 if (std::abs(a[i][j][k] - b[i][j][k]) > 1e-14)
    //                     printf("a:%.16e,b:%.16e,e:%.16e\n", a[i][j][k],
    //                            b[i][j][k], a[i][j][k] - b[i][j][k]);
    //             }
    //         }
    //     }
    // }
    // {
    //     auto a = IO::read_vector_3D<double>("eulerian_force_v.txt");
    //     auto b = IO::read_vector_3D<double>("eulerian_force_v_kokkos.txt");
    //     // 遍历三维向量
    //     for (size_t i = 0; i < a.size(); ++i)
    //     {
    //         for (size_t j = 0; j < a[i].size(); ++j)
    //         {
    //             for (size_t k = 0; k < a[i][j].size(); ++k)
    //             {
    //                 if (std::abs(a[i][j][k] - b[i][j][k]) > 1e-14)
    //                     printf("a:%.16e,b:%.16e,e:%.16e\n", a[i][j][k],
    //                            b[i][j][k], a[i][j][k] - b[i][j][k]);
    //             }
    //         }
    //     }
    // }
    // {
    //     auto a = IO::read_vector_3D<double>("eulerian_force_w.txt");
    //     auto b = IO::read_vector_3D<double>("eulerian_force_w_kokkos.txt");
    //     // 遍历三维向量
    //     for (size_t i = 0; i < a.size(); ++i)
    //     {
    //         for (size_t j = 0; j < a[i].size(); ++j)
    //         {
    //             for (size_t k = 0; k < a[i][j].size(); ++k)
    //             {
    //                 if (std::abs(a[i][j][k] - b[i][j][k]) > 1e-14)
    //                     printf("a:%.16e,b:%.16e,e:%.16e\n", a[i][j][k],
    //                            b[i][j][k], a[i][j][k] - b[i][j][k]);
    //             }
    //         }
    //     }
    // }
    // {
    //     auto a = IO::read_vector_1D<double3>("solid_velocities.txt");
    //     auto b = IO::read_vector_1D<double3>("solid_velocities_kokkos.txt");
    //     for (size_t i = 0; i < a.size(); i++)
    //     {
    //         auto error = std::abs(a[i].x - b[i].x) + std::abs(a[i].y -
    //         b[i].y)
    //                      + std::abs(a[i].z - b[i].z);
    //         if (error > 1e-15)
    //         {
    //             printf("%.20e\n", error);
    //         }
    //     }
    // }
    printScopeProfiler();
}