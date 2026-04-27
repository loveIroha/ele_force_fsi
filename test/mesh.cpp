/// @date 2023-06-14
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试网格模块
///
///

#include <MeshTools/BasicMesh2D.h>
#include <MeshTools/BuildDofmap2D.h>
#include <MeshTools/QuadratureRules.h>
#include <config.h>
#include <dolfin.h>

#include <tuple>

const int degree    = 1;
const int QUADRATIC = 2;
const int DIM       = 2;



double2 function_any(const double3& x) { return make_double2(x.x, x.y); }

template <typename T>
void print_gauss_points(T a) {
    printf("积分点数量%d\n", a.num_points());
    for (int i = 0; i < a.num_points(); i++) {
        printf("高斯积分点:    %.12e,%.12e,%.12e\n", a._points[i].x, a._points[i].y, a._points[i].z);
        printf("高斯积分权重:  %.12e\n", a._weights[i]);
    }
}

void test_gauss_points() {
    std::tuple b{
        npuheart::simplex_quadrature<double3, double, 1, 1>, npuheart::simplex_quadrature<double3, double, 3, 1>,
        npuheart::simplex_quadrature<double3, double, 5, 1>, npuheart::simplex_quadrature<double3, double, 7, 1>,
        npuheart::simplex_quadrature<double3, double, 9, 1>, npuheart::simplex_quadrature<double3, double, 1, 2>,
        npuheart::simplex_quadrature<double3, double, 3, 2>, npuheart::simplex_quadrature<double3, double, 5, 2>,
        npuheart::simplex_quadrature<double3, double, 7, 2>, npuheart::simplex_quadrature<double3, double, 9, 2>,
        npuheart::simplex_quadrature<double3, double, 1, 3>, npuheart::simplex_quadrature<double3, double, 3, 3>,
        npuheart::simplex_quadrature<double3, double, 5, 3>, npuheart::simplex_quadrature<double3, double, 7, 3>,
        npuheart::simplex_quadrature<double3, double, 9, 3>};

    print_gauss_points(std::get<0>(b));
    print_gauss_points(std::get<1>(b));
    print_gauss_points(std::get<2>(b));
    print_gauss_points(std::get<3>(b));
    print_gauss_points(std::get<4>(b));
    print_gauss_points(std::get<5>(b));
    print_gauss_points(std::get<6>(b));
    print_gauss_points(std::get<7>(b));
    print_gauss_points(std::get<8>(b));
    print_gauss_points(std::get<9>(b));
    print_gauss_points(std::get<10>(b));
    print_gauss_points(std::get<11>(b));
    print_gauss_points(std::get<12>(b));
    print_gauss_points(std::get<13>(b));
    print_gauss_points(std::get<14>(b));
}

void print_double2(double2 data, const std::string& name) { LOG_F(INFO, "%s : %f %f. ", name.c_str(), data.x, data.y); }

void print_double3(double3 data, const std::string& name) {
    LOG_F(INFO, "%s : %f %f %f. ", name.c_str(), data.x, data.y, data.z);
}

void print_double4(double4 data, const std::string& name) {
    LOG_F(INFO, "%s : %f %f %f %f. ", name.c_str(), data.x, data.y, data.z, data.w);
}

int test_quadratic(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin) {
    // 测试dolfin_build_dof是否出错
    // Coordinates for all dofs.
    std::vector<double3> vertices;
    std::vector<int4>    cells;
    std::vector<int3>    boundaries;

    // used for quadratic function defined on the mesh.
    std::vector<size_t>  dofmaps;
    std::vector<double3> dof_coordinates;

    // dolfin_build_dof<QUADRATIC, DIM>(solid_mesh_dolfin, cells, vertices,
    // dofmaps, dof_coordinates);
    BasicMesh<QUADRATIC, DIM> basic_mesh(solid_mesh_dolfin);

    // 创建、读取定义在网格上的函数
    basic_mesh.create_function<double2>("abc");
    basic_mesh.list_functions();
    auto& function = basic_mesh.find_function<double2>("abc");
    basic_mesh.set_function("abc", function_any);

    std::vector<double2> results_on_qr_points(basic_mesh.quadrature_rules.size());
    std::vector<double2> results_on_vertices(basic_mesh.vertices.size());

    basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
        function.data(), basic_mesh.dofmaps.data(), results_on_qr_points.data(), basic_mesh.num_cells());

    basic_mesh.polynomial.evaluate_vertices<double2, double>(function.data(), basic_mesh.dofmaps.data(),
                                                             basic_mesh.cells.data(), basic_mesh.vertices.data(),
                                                             results_on_vertices.data(), basic_mesh.num_cells());

    // 输出网格节点上的插值结果
    // for (size_t i = 0; i < basic_mesh.vertices.size(); i++)
    // {
    //     LOG_F(INFO, "顶点 %d 的坐标为 %f %f .", i, basic_mesh.vertices[i].x,
    //     basic_mesh.vertices[i].y); LOG_F(INFO, "顶点 %d 的值为   %f %f .", i,
    //     results_on_vertices[i].x, results_on_vertices[i].y); auto exact_value =
    //     function_any(basic_mesh.vertices[i]); LOG_F(INFO, "ERROR %d : %f %f.",
    //     exact_value.x - results_on_vertices[i].x, exact_value.y -
    //     results_on_vertices[i].y);
    // }

    auto q1 = basic_mesh.quadrature<double2, double>("abc");
    print_double2(q1, "积分结果:");

    // 计算基函数在高斯积分点的值
    basic_mesh.polynomial.calculate_basis_values();
}

int test_linear(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin) {
    // 测试dolfin_build_dof是否出错
    // Coordinates for all dofs.
    std::vector<double3> vertices;
    std::vector<int4>    cells;
    std::vector<int3>    boundaries;

    // used for quadratic function defined on the mesh.
    std::vector<size_t>  dofmaps;
    std::vector<double3> dof_coordinates;

    // dolfin_build_dof<degree, DIM>(solid_mesh_dolfin, cells, vertices, dofmaps,
    // dof_coordinates);

    // 读入BasicMesh
    // TODO: 测试dofmap是否正确
    BasicMesh<degree, DIM> basic_mesh(solid_mesh_dolfin);

    // 面积计算是否正确。
    double sum_area = 0.0;
    for (size_t i = 0; i < basic_mesh.num_cells(); i++) {
        sum_area += basic_mesh.volumes[i];
        // LOG_F(INFO, "单元 %d 的面积为 %f .", i, basic_mesh.volumes[i]);
    }
    // LOG_F(INFO, "总面积为 %f .", sum_area);

    // 创建、读取定义在网格上的函数
    basic_mesh.create_function<double2>("linear_function");
    basic_mesh.list_functions();
    auto& function = basic_mesh.find_function<double2>("linear_function");
    basic_mesh.set_function("linear_function", function_any);

    std::vector<double2> results_on_qr_points(basic_mesh.quadrature_rules.size());
    std::vector<double2> results_on_vertices(basic_mesh.vertices.size());

    basic_mesh.polynomial.evaluate_quadrature_points<double2, double>(
        function.data(), basic_mesh.dofmaps.data(), results_on_qr_points.data(), basic_mesh.num_cells());

    basic_mesh.polynomial.evaluate_vertices<double2, double>(function.data(), basic_mesh.dofmaps.data(),
                                                             basic_mesh.cells.data(), basic_mesh.vertices.data(),
                                                             results_on_vertices.data(), basic_mesh.num_cells());

    // // 输出网格节点上的插值结果
    // for (size_t i = 0; i < basic_mesh.vertices.size(); i++)
    // {
    //     LOG_F(INFO, "顶点 %d 的坐标为 %f %f .", i, basic_mesh.vertices[i].x,
    //     basic_mesh.vertices[i].y); LOG_F(INFO, "顶点 %d 的值为   %f %f .", i,
    //     results_on_vertices[i].x, results_on_vertices[i].y); auto exact_value =
    //     function_any(basic_mesh.vertices[i]); LOG_F(INFO, "ERROR %d : %f %f.",
    //     exact_value.x - results_on_vertices[i].x, exact_value.y -
    //     results_on_vertices[i].y);
    // }

    auto q1 = basic_mesh.quadrature<double2, double>("linear_function");
    print_double2(q1, "积分结果:");

    return 0;
}

// 通过位移计算移动后的固体的面积
int test_displacement(std::shared_ptr<dolfin::Mesh> solid_mesh_dolfin) {
    // 读入BasicMesh
    BasicMesh<1, 2> basic_mesh(solid_mesh_dolfin);

    // 创建、读取定义在网格上的函数
    basic_mesh.create_function<double2>("linear_function");
    basic_mesh.list_functions();
    auto& function = basic_mesh.find_function<double2>("linear_function");
    basic_mesh.set_function("linear_function", function_any);

    std::vector<double2> vertices_displacement(basic_mesh.vertices.size(), make_double2(0.0, 0.0));
    std::vector<double3> vertices_position(basic_mesh.vertices.size(), make_double3(0.0, 0.0, 0.0));

    // 计算节点上的位移
    basic_mesh.polynomial.evaluate_vertices<double2, double>(function.data(), basic_mesh.dofmaps.data(),
                                                             basic_mesh.cells.data(), basic_mesh.vertices.data(),
                                                             vertices_displacement.data(), basic_mesh.num_cells());

    // 使用 Lambda 函数对两个向量的对应元素进行相加
    std::transform(basic_mesh.vertices.begin(), basic_mesh.vertices.end(), vertices_displacement.begin(),
                   vertices_position.begin(),
                   [](double3 a, double2 b) { return make_double3(a.x + b.x, a.y + b.y, 0.0); });

    // 输出网格节点上的插值结果
    for (size_t i = 0; i < basic_mesh.vertices.size(); i++) {
        // LOG_F(INFO, "顶点 %d 的坐标为 %f %f .", i, basic_mesh.vertices[i].x,
        // basic_mesh.vertices[i].y); LOG_F(INFO, "顶点 %d 的值为   %f %f .", i,
        // vertices_displacement[i].x, vertices_displacement[i].y);
        auto exact_value = function_any(basic_mesh.vertices[i]);
        LOG_F(INFO, "ERROR %d : %f %f.", i, exact_value.x - vertices_displacement[i].x,
              exact_value.y - vertices_displacement[i].y);
    }

    std::vector<double> areas(basic_mesh.num_cells());

    basic_mesh.polynomial.triangle_area_all_cells(basic_mesh.cells.data(), vertices_position.data(), areas.data(),
                                                  basic_mesh.num_cells());
    double areas_sum          = std::accumulate(areas.begin(), areas.end(), 0.0);
    double areas_sum_eulerian = std::accumulate(basic_mesh.volumes.begin(), basic_mesh.volumes.end(), 0.0);
    LOG_F(INFO, "固体单元数：%zu,当前固体面积 %.16e, 初始固体面积 %.16e", areas.size(), areas_sum, areas_sum_eulerian);
    printf("%f\n", (areas_sum - areas_sum_eulerian) / areas_sum_eulerian);
    return areas_sum;
}
void test_1_different_mesh(int n) {
    // 读取网格
    auto mesh_file = geometry_path("temp/circle_");
    mesh_file += std::to_string(n);
    mesh_file += ".xdmf";
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();

    // test_linear(solid_mesh_dolfin);

    test_quadratic(solid_mesh_dolfin);
}

void test_1_unitsquare_mesh(int n) {
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>(dolfin::UnitSquareMesh(n, n, "left"));
    // test_linear(solid_mesh_dolfin);
    test_quadratic(solid_mesh_dolfin);
    // test_displacement(solid_mesh_dolfin);
}

int main() {
    // 测试高斯积分点
    // test_gauss_points();

    // test_1_1();
    // std::vector<int> mesh_size{5, 10, 20, 30, 40};
    std::vector<int> mesh_size{5};
    // std::vector<int> mesh_size{1};

    for (size_t i = 0; i < mesh_size.size(); i++) {
        // test_1_different_mesh(mesh_size[i]);
        test_1_unitsquare_mesh(mesh_size[i]);
    }

    return 0;
}
