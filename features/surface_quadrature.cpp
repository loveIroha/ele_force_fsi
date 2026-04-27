/// @date 2023-04-22
/// @file surface_quadrature.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 三维空间中，三角形经过仿射变换后，计算三角形内部点的坐标。
///
///
#include <cmath>
#include <iostream>
#include <limits>

template <typename T>
struct Point {
    T x = T{};
    T y = T{};
    T z = T{};
};

template <typename T>
bool is_points_coplanar(T x1, T y1, T z1, T x2, T y2, T z2, T x3, T y3, T z3, T x4, T y4, T z4) {
    T eps = std::numeric_limits<T>::epsilon(); // 误差阈值
    T ax  = x2 - x1;
    T ay  = y2 - y1;
    T az  = z2 - z1;
    T bx  = x3 - x1;
    T by  = y3 - y1;
    T bz  = z3 - z1;
    T cx  = x4 - x1;
    T cy  = y4 - y1;
    T cz  = z4 - z1;
    T nx  = ay * bz - az * by;
    T ny  = az * bx - ax * bz;
    T nz  = ax * by - ay * bx;
    T len = nx * cx + ny * cy + nz * cz;
    return std::abs(len) < eps;
}

// NOTE: 关于有限元单元仿射变化的内容可以参考何晓明第二节课的PPT
// NOTE: 三维空间中的三角形单元需要先投影到坐标平面中再进行求解
template <typename T>
bool affine_coordinate(const Point<T>& p1, const Point<T>& p2, const Point<T>& p3, const Point<T>& p4, double& alpha,
                       double& beta) {
    if (!is_points_coplanar(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, p4.x, p4.y, p4.z)) {
        printf("Not all points are on the same plane.\n");
        return false;
    }

    T det = T{};
    T eps = std::numeric_limits<T>::epsilon();

    det = (p2.x - p1.x) * (p3.z - p1.z) - (p3.x - p1.x) * (p2.z - p1.z);
    if (std::abs(det) > eps) {
        printf("projected onto the XZ plane, J = %lf\n", det);
        alpha = ((p3.z - p1.z) * (p4.x - p1.x) - (p3.x - p1.x) * (p4.z - p1.z)) / det;
        beta  = (-(p2.z - p1.z) * (p4.x - p1.x) + (p2.x - p1.x) * (p4.z - p1.z)) / det;
        return true;
    }

    det = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
    if (std::abs(det) > eps) {
        printf("projected onto the XY plane, J = %lf\n", det);
        alpha = ((p3.y - p1.y) * (p4.x - p1.x) - (p3.x - p1.x) * (p4.y - p1.y)) / det;
        beta  = (-(p2.y - p1.y) * (p4.x - p1.x) + (p2.x - p1.x) * (p4.y - p1.y)) / det;
        return true;
    }

    det = (p2.y - p1.y) * (p3.z - p1.z) - (p3.y - p1.y) * (p2.z - p1.z);
    if (std::abs(det) > eps) {
        printf("projected onto the YZ plane, J = %lf\n", det);
        alpha = ((p3.z - p1.z) * (p4.y - p1.y) - (p3.y - p1.y) * (p4.z - p1.z)) / det;
        beta  = (-(p2.z - p1.z) * (p4.y - p1.y) + (p2.y - p1.y) * (p4.z - p1.z)) / det;
        return true;
    }

    return false;
}

// P1P4 = \alpha P1P2+\beta P1P3
template <typename T>
bool local_coordinate(const Point<T>& p1, const Point<T>& p2, const Point<T>& p3, Point<T>& p4, double& alpha,
                      double& beta) {
    p4.x = alpha * (p2.x - p1.x) + beta * (p3.x - p1.x) + p1.x;
    p4.y = alpha * (p2.y - p1.y) + beta * (p3.y - p1.y) + p1.y;
    p4.z = alpha * (p2.z - p1.z) + beta * (p3.z - p1.z) + p1.z;
}

// Point<double> p1{0.0, 0.0, 0.0};
// Point<double> p2{1.0, 0.0, 0.0};
// Point<double> p3{0.0, 1.0, 0.0};
// Point<double> p4{0.5, 0.5, 0.0};

// Point<double> p1{0.0, 0.0, 0.0};
// Point<double> p2{1.0, 0.0, 0.0};
// Point<double> p3{0.0, 0.0, 1.0};
// Point<double> p4{0.5, 0.5, 0.5};

// Point<double> p1{0.0, 0.0, 0.0};
// Point<double> p2{0.0, 1.0, 0.0};
// Point<double> p3{0.0, 0.0, 1.0};
// Point<double> p4{0.0, 0.5, 0.5};

Point<double> p1{3.0, 0.0, 0.0};
Point<double> p2{0.0, 3.0, 0.0};
Point<double> p3{0.0, 0.0, 6.0};
Point<double> p4{1.0, 1.0, 2.0};

Point<double> p1_{4.0, 0.0, 0.0};
Point<double> p2_{1.0, 3.0, 0.0};
Point<double> p3_{1.0, 0.0, 3.0};
Point<double> p4_;

double alpha, beta;

const double Gauss_coefficient_reference_triangle_4[4]
    = {0.052831216351297, 0.052831216351297, 0.197168783648703, 0.197168783648703};

const double Gauss_point_reference_triangle_4[8]
    = {0.788675134594813, 0.166666666666667, 0.788675134594813, 0.044658198738520,
       0.211324865405187, 0.622008467928146, 0.211324865405187, 0.166666666666667};

const double Gauss_coefficient_reference_triangle_9[9]
    = {0.098765432098765, 0.008696116155807, 0.008696116155807, 0.068464377671354, 0.068464377671354,
       0.061728395061728, 0.061728395061728, 0.013913785849291, 0.109543004274166};

const double Gauss_point_reference_triangle_9[18] = {
    0.500000000000000, 0.250000000000000, 0.887298334620742, 0.100000000000000, 0.887298334620742, 0.012701665379258,
    0.112701665379258, 0.787298334620742, 0.112701665379258, 0.100000000000000, 0.500000000000000, 0.443649167310371,
    0.500000000000000, 0.056350832689629, 0.887298334620742, 0.056350832689629, 0.112701665379258, 0.443649167310371};

const double Gauss_coefficient_reference_triangle_3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};

const double Gauss_point_reference_triangle_3[6] = {0.500000000000000, 0.000000000000000, 0.500000000000000,
                                                    0.500000000000000, 0.000000000000000, 0.500000000000000};

int main() {
    // 参考四面体单元
    Point<double> tetrahedron[4] = {Point<double>{0.0, 0.0, 0.0}, Point<double>{1.0, 0.0, 0.0},
                                    Point<double>{0.0, 1.0, 0.0}, Point<double>{0.0, 0.0, 1.0}};

    // 三角形单元的局部索引，三角形对面的顶点为索引的顺序。
    int local_index = 0;

    // triangle为参考四面体单元上的四个三角形，其中triangle_ref是参考三角形单元
    Point<double> triangle_ref[3] = {tetrahedron[0], tetrahedron[1], tetrahedron[2]};
    Point<double> triangle[3]     = {tetrahedron[0], tetrahedron[1], tetrahedron[2]};
    if (local_index < 3) { triangle[local_index] = tetrahedron[3]; }

    // 高斯积分点
    const double* gauss_points = Gauss_point_reference_triangle_9;

    // NOTE: 坐标变换的顺序为参考三角形单元 -> 参考四面体单元 -> 局部四面体单元
    // NOTE: 需要两个函数来实现
    // 计算每个高斯积分点在四面体单元上的坐标
    for (size_t i = 0; i < 9; i++) {
        Point<double> p{gauss_points[i * 2], gauss_points[i * 2 + 1]};
        Point<double> p_;
        double        alpha, beta;
        affine_coordinate(triangle_ref[0], triangle_ref[1], triangle_ref[2], p, alpha, beta);
        local_coordinate(triangle[0], triangle[1], triangle[2], p_, alpha, beta);
        printf("Alpha and beta: %lf,%lf\n", alpha, beta);
        printf("Old point: %lf,%lf,%lf\n", p.x, p.y, p.z);
        printf("New point: %lf,%lf,%lf\n", p_.x, p_.y, p_.z);
    }

    // TODO: 计算每个高斯积分点在局部单元上的坐标
    Point<double> triangle_local[3];
    // 上面计算出来的p_作为参考单元上的高斯积分点

    return 0;
}