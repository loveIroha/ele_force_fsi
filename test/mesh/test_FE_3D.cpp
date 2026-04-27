/// @date 2024-01-27
/// @file test_FE_3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///
#include <nlohmann/json.hpp>

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BasicMesh.h>
#include <config.h>
#include <dolfin/generation/UnitCubeMesh.h>
#include <dolfin/io/XDMFFile.h>
#include <io.h>

// TODO: 这个文件需要参考其他程序库，然后大改

// TODO: 测试积分
// TODO: 测试其他网格

// TEST 1: 测试 evaluate 相关函数的正确性
// 1. 定义网格上的线性向量函数 f(x,y,z) = (x,y,z)
// 2. 调用 evaluate_vertices 函数在网格节点处的值



// double3              function_linear(const double3& x) { return make_double3(x.x, x.y, x.z); }
// std::vector<double3> function_grad(const double3& x) {
//     std::vector<double3> result
//         = {make_double3(1, 0, 0), make_double3(0, 1, 0), make_double3(0, 0, 1)};
//     return result;
// }
double3              function_linear(const double3& x) { return make_double3(x.x * x.x, x.y * x.y, x.z * x.z); }
std::vector<double3> function_grad(const double3& x) {
    std::vector<double3> result
        = {make_double3(2 * x.x, 0, 0), make_double3(0, 2 * x.y, 0), make_double3(0, 0, 2 * x.z)};
    return result;
}
double3              function_linear(const double4& x) { return function_linear(make_double3(x.x, x.y, x.z)); }
std::vector<double3> function_grad(const double4& x) { return function_grad(make_double3(x.x, x.y, x.z)); }


template <int FE_ORDER>
void test_evaluate_vertices(std::shared_ptr<BasicMesh<FE_ORDER>> solid_mesh) {
    ScopeProfiler _{__func__};

    // 创建、读取定义在网格上的函数
    solid_mesh->template create_function<double3>("linear");
    solid_mesh->template list_functions();
    auto& quadrature_rules = solid_mesh->get_quadrature_rules();
    auto& function         = solid_mesh->template find_function<double3>("linear");
    solid_mesh->set_function("linear", function_linear);

    std::vector<double3> results_on_qr_points(quadrature_rules.size());
    std::vector<double3> results_on_vertices(solid_mesh->get_vertices().size());

    solid_mesh->polynomial->template evaluate_quadrature_points<double3, double>(
        function.data(), solid_mesh->dofmaps.data(), results_on_qr_points.data(), solid_mesh->num_cells());

    // 1. 函数在高斯积分点上的插值结果
    for (size_t i = 0; i < quadrature_rules.size(); i++) {
        auto exact_value = function_linear(quadrature_rules[i]);
        // LOG_F(INFO, "积分点 %d 的坐标为 %f %f %f.", i, quadrature_rules[i].x, quadrature_rules[i].y,
        //       quadrature_rules[i].z);
        // LOG_F(INFO, "积分点 %d 的值为   %f %f %f.", i, results_on_qr_points[i].x, results_on_qr_points[i].y,
        //       results_on_qr_points[i].z);
        // LOG_F(INFO, "ERROR %d : %f %f %f.", i, exact_value.x - results_on_qr_points[i].x,
        //       exact_value.y - results_on_qr_points[i].y, exact_value.z - results_on_qr_points[i].z);
        // printf("%.4e\n", algebra::distance(exact_value, results_on_qr_points[i]));
        // CHECK_F(algebra::distance(exact_value, results_on_qr_points[i]) < NPUHEART_EPS_LARGE * 10,
        //         "evaluate_vertices error");
    }

    solid_mesh->polynomial->template evaluate_vertices<double3, double>(
        function.data(), solid_mesh->dofmaps.data(), solid_mesh->cells.data(), solid_mesh->vertices.data(),
        results_on_vertices.data(), solid_mesh->num_cells());

    // 2. 函数在网格节点上的插值结果
    for (size_t i = 0; i < solid_mesh->vertices.size(); i++) {
        auto exact_value = function_linear(solid_mesh->vertices[i]);
        // LOG_F(INFO, "顶点 %d 的坐标为 %f %f %f.", i, solid_mesh->vertices[i].x, solid_mesh->vertices[i].y,
        //       solid_mesh->vertices[i].z);
        // LOG_F(INFO, "顶点 %d 的值为   %f %f %f.", i, results_on_vertices[i].x, results_on_vertices[i].y,
        //       results_on_vertices[i].z);
        // LOG_F(INFO, "ERROR %d : %f %f %f.", i, exact_value.x - results_on_vertices[i].x,
        //       exact_value.y - results_on_vertices[i].y, exact_value.z - results_on_vertices[i].z);
        // CHECK_F(algebra::distance(exact_value, results_on_vertices[i]) < NPUHEART_EPS_LARGE, "evaluate_vertices error");
    }
}

template <int FE_ORDER>
void test_evaluate_grad_vertices(std::shared_ptr<BasicMesh<FE_ORDER>> solid_mesh) {
    ScopeProfiler _{__func__};

    // 创建、读取定义在网格上的函数
    solid_mesh->template create_function<double3>("linear");
    solid_mesh->template list_functions();
    auto& quadrature_rules = solid_mesh->get_quadrature_rules();
    auto& function         = solid_mesh->template find_function<double3>("linear");
    solid_mesh->set_function("linear", function_linear);
    std::vector<double3> results_on_qr_points_grad(quadrature_rules.size() * 3);
    solid_mesh->polynomial->template evaluate_grad_quadrature_points<double3, double>(
        function.data(), solid_mesh->dofmaps.data(), solid_mesh->inv_Hs.data(), results_on_qr_points_grad.data(),
        solid_mesh->num_cells());
    
    // 函数的导数在网格节点上的插值结果
    for (size_t i = 0; i < quadrature_rules.size(); i++) {
        auto exact_value = function_grad(quadrature_rules[i]);
        // LOG_F(INFO, "顶点 %d 的坐标为 %f %f %f.", i, solid_mesh->vertices[i].x, solid_mesh->vertices[i].y,
        //       solid_mesh->vertices[i].z);
        // LOG_F(INFO, "顶点 %d 的值为   %f %f %f.", i, results_on_qr_points_grad[3 * i + 0].x,
        // results_on_qr_points_grad[3 * i + 0].y,
        //       results_on_qr_points_grad[3 * i + 0].z);
        // LOG_F(INFO, "顶点 %d 的值为   %f %f %f.", i, results_on_qr_points_grad[3 * i + 1].x,
        // results_on_qr_points_grad[3 * i + 1].y,
        //       results_on_qr_points_grad[3 * i + 1].z);
        // LOG_F(INFO, "顶点 %d 的值为   %f %f %f.", i, results_on_qr_points_grad[3 * i + 2].x,
        // results_on_qr_points_grad[3 * i + 2].y,
        //       results_on_qr_points_grad[3 * i + 2].z);
        // LOG_F(INFO, "ERROR %d : %.4e %.4e %.4e.", i, exact_value[0].x - results_on_qr_points_grad[3 * i + 0].x,
        //       exact_value[0].y - results_on_qr_points_grad[3 * i + 0].y,
        //       exact_value[0].z - results_on_qr_points_grad[3 * i + 0].z);
        // CHECK_F(algebra::distance(exact_value, results_on_vertices[i]) < NPUHEART_EPS_LARGE, "evaluate_vertices
        // error");
    }

}


template <int FE_ORDER>
double test_calculate_volume(std::shared_ptr<BasicMesh<FE_ORDER>> solid_mesh){

    ScopeProfiler _{__func__};

    // 创建、读取定义在网格上的函数
    solid_mesh->template create_function<double3>("linear");
    solid_mesh->template list_functions();
    auto& quadrature_rules = solid_mesh->get_quadrature_rules();
    auto& _solid_displacement = solid_mesh->template find_function<double3>("linear");
    solid_mesh->set_function("linear", function_linear);
    double volumes_sum = solid_mesh->current_area(_solid_displacement);
    return volumes_sum;
}
// 读取真实左心室网格
template <int FE_ORDER>
auto read_real_LV_mesh() {
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    dolfin::XDMFFile mesh_file_1(geometry_path("/mnt/large2/gjh/realistic_left_ventricle/mesh_scale.xdmf"));
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto solid_mesh = std::make_shared<BasicMesh<FE_ORDER>>(solid_mesh_dolfin);
    return solid_mesh;
}
template <int FE_ORDER>
auto create_box_mesh(size_t n = 1) {
    ScopeProfiler _{__func__};
    dolfin::Point p0(0.0, 0.0, 0.0);
    dolfin::Point p1(1.0, 1.0, 1.0);
    auto          solid_mesh_dolfin = std::make_shared<dolfin::Mesh>(
        dolfin::BoxMesh::create({p0, p1}, {n, n, n}, dolfin::CellType::Type::tetrahedron));
    auto solid_mesh = std::make_shared<BasicMesh<FE_ORDER>>(solid_mesh_dolfin);
    return solid_mesh;
}

// CASE 1: 在真实左心室网格上测试
void test_evaluate_vertices_all() {
    {
        // 有限元阶数：1
        printf("FE_ORDER = 1\n");
        // TEST 1: 测试 evaluate_vertices 函数
        auto mesh = read_real_LV_mesh<1>();
        test_evaluate_vertices<1>(mesh);
        test_evaluate_grad_vertices<1>(mesh);
        // TEST 2: 测试 calculate_volume 函数
        auto volumes_sum = test_calculate_volume<1>(mesh);
        CHECK_F(std::abs(volumes_sum-3.3024669135616400e+05)/3e+05 < NPUHEART_EPS_LARGE, "calculate_volume error");
    }
    {
        // 有限元阶数：2
        printf("FE_ORDER = 2\n");
        // TEST 1: 测试 evaluate_vertices 函数
        auto mesh = read_real_LV_mesh<2>();
        test_evaluate_vertices<2>(mesh);
        test_evaluate_grad_vertices<2>(mesh);
        // TEST 2: 测试 calculate_volume 函数
        auto volumes_sum = test_calculate_volume<2>(mesh);
        CHECK_F(std::abs(volumes_sum-3.3024669135616830e+05)/3e+05 < NPUHEART_EPS_LARGE, "calculate_volume error");
    } {
        // printf("BOX, FE_ORDER = 1\n");
        // size_t n = 4;
        // auto mesh = create_box_mesh<1>(4);
        // test_evaluate_vertices<1>(mesh);
    }
    {
        // printf("BOX, FE_ORDER = 2\n");
        // size_t n = 4;
        // auto mesh = create_box_mesh<2>();
        // test_evaluate_vertices<2>(mesh);
    }
    {

        // 在...网格上测试
    } {
        // 在...网格上测试
    }
}

int main() {
    // TEST 1: 测试 evaluate   相关函数的正确性
    test_evaluate_vertices_all();
    // TEST 2: 测试 integral   相关函数的正确性
    // TEST 3: 测试 derivative 相关函数的正确性
   printScopeProfiler();
    return 0;
}
