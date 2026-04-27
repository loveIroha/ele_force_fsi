/// @date 2023-06-14
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试网格模块
///        1. 测试延拓算子
///        2. 测试插值算子

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <MeshTools/BuildDofmap2D.h>
#include <MeshTools/QuadratureRules.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction2D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven2D/DiskDriven.h>
#include <config.h>
#include <dolfin.h>
#include <io/vector_io.h>
#include <io/writeVTK.h>

#include <tuple>

const int degree = 1;
const int DIM    = 2;



double2 function_any(const double3& x) { return make_double2(x.y, x.x); }

double function_any_source(const double3& x) { return x.y; }

int test_distribution(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<2> domain_mesh) {
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "延拓算子");
    // 二维三角形网格、一阶多项式
    BasicMesh<degree, DIM> basic_mesh(solid_mesh_dolfin);

    // 在固体网格上定义向量函数
    basic_mesh.create_function<double2>("1");
    basic_mesh.list_functions();
    auto& function = basic_mesh.find_function<double2>("1");
    basic_mesh.set_function("1", function_any);
    std::vector<double2> function_qr(basic_mesh.quadrature_rules.size());

    // 在固体网格上定义标量函数
    auto& function_source = basic_mesh.create_function<double>("source");
    basic_mesh.set_function("source", function_any_source);
    std::vector<double> function_source_qr(basic_mesh.quadrature_rules.size());

    // 计算函数在高斯积分点上的值
    basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(function.data(), basic_mesh.dofmaps.data(),
                                                                      function_qr.data(), basic_mesh.num_cells());
    basic_mesh.polynomial.evaluate_quadrature_points<double, double>(function_source.data(), basic_mesh.dofmaps.data(),
                                                                     function_source_qr.data(), basic_mesh.num_cells());

    // 延拓算子
    std::vector<double> eulerian_u(domain_mesh.get_size_u());
    std::vector<double> eulerian_v(domain_mesh.get_size_v());
    std::vector<double> eulerian_s(domain_mesh.get_size_p());

    // 交错网格
    basic_mesh.distribute_force(eulerian_u, eulerian_v, function_qr, basic_mesh.quadrature_rules,
                                make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_u(),
                                domain_mesh.get_dim_v());

    // 网格单元中心
    basic_mesh.distribute_source(eulerian_s, function_source_qr, basic_mesh.quadrature_rules,
                                 make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_p());

    // 检查延拓算子的参数
    LOG_F(INFO, "eulerian_s size: %zu", eulerian_s.size());
    LOG_F(INFO, "function_source_qr size: %zu", function_source_qr.size());
    LOG_F(INFO, "basic_mesh.quadrature_rules size: %zu", basic_mesh.quadrature_rules.size());
    LOG_F(INFO, "function_qr size: %zu", function_qr.size());

    for (int i = 0; i < function_source_qr.size(); i++) {
        LOG_F(INFO, "function_source_qr: %d %f", i, function_source_qr[i]);
        LOG_F(INFO, "quadrature_rules: %d %f %f %f %.30f", i, basic_mesh.quadrature_rules[i].x,
              basic_mesh.quadrature_rules[i].y, basic_mesh.quadrature_rules[i].z, basic_mesh.quadrature_rules[i].w);
        LOG_F(INFO, "error: %d %.30f", i, function_source_qr[i] - basic_mesh.quadrature_rules[i].y);
    }

    auto dim_u = domain_mesh.get_dim_u();
    auto dim_v = domain_mesh.get_dim_v();

    // TODO : distribution 的误差挺大的
    for (int i = 0; i < dim_u.x; i++) {
        for (int j = 1; j < dim_u.y - 1; j++) {
            double x = (i)*domain_mesh._dx;
            double y = (j - 0.5) * domain_mesh._dy;

            auto exact = function_any(make_double3(x, y, 0.0));
            if ((x - 0.6) * (x - 0.6) + (y - 0.5) * (y - 0.5) < 0.2 * 0.2) {
                eulerian_u[i + j * dim_u.x] = eulerian_u[i + j * dim_u.x] - exact.x;
                LOG_F(INFO, "distribution error u: %.20f ", eulerian_u[i + j * dim_u.x] - exact.x);
            } else {
                eulerian_u[i + j * dim_u.x] = 0.0;
            }
        }
    }

    for (int i = 1; i < dim_v.x - 1; i++) {
        for (int j = 0; j < dim_v.y; j++) {
            double x     = (i - 0.5) * domain_mesh._dx;
            double y     = j * domain_mesh._dy;
            auto   exact = function_any(make_double3(x, y, 0.0));
            if ((x - 0.6) * (x - 0.6) + (y - 0.5) * (y - 0.5) < 0.05 * 0.05)
                LOG_F(INFO, "distribution error v: %.20f ", eulerian_v[i + j * dim_v.x] - exact.y);
        }
    }

    // 输出延拓算子作用后的结果
    std::string filename_u  = "results/function_u.vti";
    std::string arrayname_u = "function_u";
    write_vtk(domain_mesh.get_dim_u(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
              make_double2(domain_mesh._dx, domain_mesh._dy), eulerian_u, filename_u, arrayname_u);

    std::string filename_v  = "results/function_v.vti";
    std::string arrayname_v = "function_v";
    write_vtk(domain_mesh.get_dim_v(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
              make_double2(domain_mesh._dx, domain_mesh._dy), eulerian_v, filename_v, arrayname_v);

    std::string filename_s  = "results/function_s.vti";
    std::string arrayname_s = "function_s";
    write_vtk(domain_mesh.get_dim_p(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
              make_double2(domain_mesh._dx, domain_mesh._dy), eulerian_s, filename_s, arrayname_s);

    dolfin::File("b.pvd") << *solid_mesh_dolfin;
    return 0;
}

int test_interpolation(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<DIM> domain_mesh) {
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "插值算子");

    // 二维三角形网格、一阶多项式
    BasicMesh<degree, DIM> basic_mesh(solid_mesh_dolfin);

    // 在固体网格上定义函数
    basic_mesh.create_function<double2>("1");
    auto&                function = basic_mesh.find_function<double2>("1");
    std::vector<double2> function_qr(basic_mesh.quadrature_rules.size());

    // 给eulerian_u和eulerian_v赋值
    [[maybe_unused]] std::vector<double> eulerian_u(domain_mesh.get_size_u());
    std::vector<double>                  eulerian_v(domain_mesh.get_size_v());
    auto                                 dim_u = domain_mesh.get_dim_u();
    auto                                 dim_v = domain_mesh.get_dim_v();
    for (int i = 0; i < dim_u.x; i++) {
        for (int j = 1; j < dim_u.y - 1; j++) {
            double x = (i)*domain_mesh._dx;
            double y = (j - 0.5) * domain_mesh._dy;

            auto exact                  = function_any(make_double3(x, y, 0.0));
            eulerian_u[i + j * dim_u.x] = exact.x;
        }
    }

    for (int i = 1; i < dim_v.x - 1; i++) {
        for (int j = 0; j < dim_v.y; j++) {
            double x                    = (i - 0.5) * domain_mesh._dx;
            double y                    = (j)*domain_mesh._dy;
            auto   exact                = function_any(make_double3(x, y, 0.0));
            eulerian_v[i + j * dim_v.x] = exact.y;
        }
    }

    // 计算高斯积分点上的函数值
    basic_mesh.interpolate_velocity(eulerian_u, eulerian_v, function_qr, basic_mesh.quadrature_rules,
                                    make_double2(domain_mesh._dx, domain_mesh._dy), domain_mesh.get_dim_u(),
                                    domain_mesh.get_dim_v());

    // 检查高斯积分点上的积分结果
    auto data1 = basic_mesh.quadrature_rules;
    auto data2 = function_qr;
    assert(data1.size() == data2.size());
    for (size_t logf_i = 0; logf_i < data1.size(); logf_i++) {
        // LOG_F(INFO, "gauss quadrature values: %d %.16e %.16e.", logf_i,
        // data1[logf_i].x, data1[logf_i].y); LOG_F(INFO, "function values : %d
        // %.16e %.16e.", logf_i, data2[logf_i].x, data2[logf_i].y);
        LOG_F(INFO, "error                  : %zu %.20e %.20e.", logf_i, data2[logf_i].y - data1[logf_i].x,
              data2[logf_i].x - data1[logf_i].y);
    }

    // 创建Dolfin变量
    dolfin::DiskDriven disk_driven(basic_mesh._mesh, 1.0);
    auto               dolfin_x = std::make_shared<dolfin::Function>(disk_driven.V);
    auto               dolfin_b = std::make_shared<dolfin::Function>(disk_driven.V);
    auto               dolfin_A = std::make_shared<dolfin::Matrix>(disk_driven.A);

    // 组装右端项
    LOG_F(INFO, "Interpolation operator.");
    std::vector<double2> rhs(basic_mesh.num_dofs());
    basic_mesh.assemble_rhs_with_values_on_quadrature(function_qr, rhs);
    dolfin_b->vector()->set_local(algebra::flatten<double2, double>(rhs));

    dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");

    // 通过dolfin输出数据
    LOG_F(INFO, "Save the results.");
    dolfin::File file("a.pvd");

    dolfin::Array<double> values(2);
    dolfin::Array<double> x(2);
    auto                  cooridnates = basic_mesh._mesh->coordinates();

    // for (size_t i = 0; i < cooridnates.size() / 2; i++)
    // {
    //     x[0] = cooridnates[i * 2 + 0];
    //     x[1] = cooridnates[i * 2 + 1];
    //     dolfin_x->eval(values, x);
    //     auto exact = function_any(make_double3(x[0], x[1], 0.0));
    //     LOG_F(INFO, "error      : %.20f %.20f", exact.x - x[1], exact.y -
    //     x[0]); LOG_F(INFO, "cooridnates: %.20f %.20f", cooridnates[i * 2 + 0],
    //     cooridnates[i * 2 + 1]); LOG_F(INFO, "values     : %.20f %.20f",
    //     values[0], values[1]);
    // }

    // printf("%f\n",error_l1);

    file << *dolfin_x;

    return 0;
}

void test_1_2() {
    LOG_SCOPE_FUNCTION(INFO);

    // 读取固体网格
    auto             mesh_file = geometry_path("temp/circle_30.xdmf");
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();

    // auto solid_mesh_dolfin =
    // std::make_shared<dolfin::Mesh>(dolfin::UnitSquareMesh(2,2));

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(10,         // Nt
                                    {16, 16},   // Nx Ny
                                    {1.0, 1.0}, // Lx Ly
                                    1.0,        // T
                                    1.0,        // rho
                                    0.01        // mu
    );

    // test_distribution(solid_mesh_dolfin, domain_mesh);
    test_interpolation(solid_mesh_dolfin, domain_mesh);
}

void test_3(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<DIM> domain_mesh) {
    // 给eulerian_u和eulerian_v赋值
    std::vector<double> eulerian_u(domain_mesh.get_size_u());
    std::vector<double> eulerian_v(domain_mesh.get_size_v());
    auto                dim_u = domain_mesh.get_dim_u();
    auto                dim_v = domain_mesh.get_dim_v();
    for (int i = 0; i < dim_u.x; i++) {
        for (int j = 1; j < dim_u.y - 1; j++) {
            double x = (i)*domain_mesh._dx;
            double y = (j - 0.5) * domain_mesh._dy;

            auto exact                  = function_any(make_double3(x, y, 0.0));
            eulerian_u[i + j * dim_u.x] = exact.x;
        }
    }
    for (int i = 1; i < dim_v.x - 1; i++) {
        for (int j = 0; j < dim_v.y; j++) {
            double x                    = (i - 0.5) * domain_mesh._dx;
            double y                    = (j)*domain_mesh._dy;
            auto   exact                = function_any(make_double3(x, y, 0.0));
            eulerian_v[i + j * dim_v.x] = exact.y;
        }
    }
    double  result[2];
    double4 qr[2];
    qr[0] = {0.3333333333333333333333, 0.22, 0, 0};
    qr[1] = {0.12, 0.23, 0, 0};

    interpolate_u(result, eulerian_u.data(), 2, dim_u, domain_mesh._dx, domain_mesh._dy, qr);
    printf("%.20f %.20f\n", result[0], result[1]);

    interpolate_v(result, eulerian_v.data(), 2, dim_v, domain_mesh._dx, domain_mesh._dy, qr);
    printf("%.20f %.20f\n", result[0], result[1]);
}

void test_4(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin, BackgroundMesh2D<DIM> domain_mesh) {
    // 给eulerian_u和eulerian_v赋值
    std::vector<double> eulerian_u(domain_mesh.get_size_u());
    std::vector<double> eulerian_v(domain_mesh.get_size_v());
    auto                dim_u = domain_mesh.get_dim_u();
    auto                dim_v = domain_mesh.get_dim_v();

    double  input[2] = {1, 2};
    double4 qr[2];
    qr[0] = {0.3333333333333333333333, 0.22, 0, 1};
    qr[1] = {0.12, 0.23, 0, 1};

    distribute_u(input, eulerian_u.data(), 1, dim_u, domain_mesh._dx, domain_mesh._dy, qr);

    std::string filename = "a.txt";
    IO::write_vector_2D(algebra::ripple(eulerian_u, dim_u.x), filename);
}

void test_2_1() {
    LOG_SCOPE_FUNCTION(INFO);

    // 读取固体网格
    auto             mesh_file = geometry_path("temp/circle_30.xdmf");
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();

    // auto solid_mesh_dolfin =
    // std::make_shared<dolfin::Mesh>(dolfin::RectangleMesh(dolfin::Point(0.5,
    // 0.4), dolfin::Point(0.7, 0.6), 10, 10));

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(10,         // Nt
                                    {50, 50},   // Nx Ny
                                    {1.0, 1.0}, // Lx Ly
                                    1.0,        // T
                                    1.0,        // rho
                                    0.01        // mu
    );

    test_distribution(solid_mesh_dolfin, domain_mesh);
    // test_interpolation(solid_mesh_dolfin, domain_mesh);
    // test_4(solid_mesh_dolfin, domain_mesh);
    test_3(solid_mesh_dolfin, domain_mesh);
}

int main() {
    LOG_SCOPE_FUNCTION(INFO);
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("warning.log", loguru::Append, loguru::Verbosity_WARNING);
    // test_1_2();
    test_2_1();
}
