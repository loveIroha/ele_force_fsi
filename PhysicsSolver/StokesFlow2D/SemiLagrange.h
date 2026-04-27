/// @date 2023-06-24
/// @file SemiLagrange.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 半拉格朗日方法处理对流项
///
///

#ifndef __SEMI_LAGRANGE__
#define __SEMI_LAGRANGE__

#include <AlgebraSolver/algebra.h>
#include <GPU/double_math.h>
#include <vector_types.h>

#include <vector>

namespace semi_lagrange {

// 在当前位置计算出发点的速度
void trace(std::vector<std::vector<double2>>& eulerian_u_d, std::vector<std::vector<double2>>& eulerian_v_d,
           const std::vector<std::vector<double>>& eulerian_u, const std::vector<std::vector<double>>& eulerian_v,
           double dt, double hx, double hy, size_t Nx, size_t Ny) {
    for (size_t i = 0; i < Nx + 1; i++) {
        for (size_t j = 1; j < Ny + 1; j++) {
            // LOG_F(INFO, "the same as printf :  %d %d %d %d .", i, j, Nx, Ny);
            eulerian_u_d[i][j].x = eulerian_u[i][j];
            eulerian_u_d[i][j].y
                = 0.25 * (eulerian_v[i][j] + eulerian_v[i][j - 1] + eulerian_v[i + 1][j] + eulerian_v[i + 1][j - 1]);
        }
        eulerian_u_d[i][0].x = eulerian_u[i][0];
        eulerian_u_d[i][0].y = 0.5 * (eulerian_v[i][0] + eulerian_v[i + 1][0]);

        eulerian_u_d[i][Ny + 1].x = eulerian_u[i][Ny + 1];
        eulerian_u_d[i][Ny + 1].y = 0.5 * (eulerian_v[i][Ny] + eulerian_v[i + 1][Ny]);
    }

    for (size_t i = 1; i < Nx + 1; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            eulerian_v_d[i][j].y = eulerian_v[i][j];
            eulerian_v_d[i][j].x
                = 0.25 * (eulerian_u[i][j] + eulerian_u[i][j + 1] + eulerian_u[i - 1][j] + eulerian_u[i - 1][j + 1]);
            // left and right ghost points
            eulerian_v_d[0][j].y = eulerian_v[0][j];
            eulerian_v_d[0][j].x = 0.5 * (eulerian_u[0][j] + eulerian_u[0][j + 1]);

            eulerian_v_d[Nx + 1][j].y = eulerian_v[Nx + 1][j];
            eulerian_v_d[Nx + 1][j].x = 0.5 * (eulerian_u[Nx][j] + eulerian_u[Nx][j + 1]);
        }
    }
}

// 计算当前位置
void current(std::vector<std::vector<double2>>& position_u_i, std::vector<std::vector<double2>>& position_v_i,
             double dt, double hx, double hy, size_t Nx, size_t Ny) {
    for (size_t i = 0; i < Nx + 1; i++) {
        for (size_t j = 0; j < Ny + 2; j++) {
            position_u_i[i][j].x = i * hx;
            position_u_i[i][j].y = (j - 0.5) * hy;
        }
    }
    for (size_t i = 0; i < Nx + 2; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            position_v_i[i][j].x = (i - 0.5) * hx;
            position_v_i[i][j].y = j * hy;
        }
    }
}

// 根据速度计算出发点的位置
void originated(const std::vector<std::vector<double2>>& eulerian_u_d,
                const std::vector<std::vector<double2>>& eulerian_v_d, std::vector<std::vector<double2>>& position_u_d,
                std::vector<std::vector<double2>>& position_v_d, const std::vector<std::vector<double2>>& position_u,
                const std::vector<std::vector<double2>>& position_v, double dt, double hx, double hy, size_t Nx,
                size_t Ny) {
    for (size_t i = 0; i < Nx + 1; i++) {
        for (size_t j = 1; j < Ny + 1; j++) {
            position_u_d[i][j] = position_u[i][j] - dt * eulerian_u_d[i][j];
        }
        position_u_d[i][0]      = position_u[i][0];
        position_u_d[i][Ny + 1] = position_u[i][Ny + 1];
    }

    for (size_t i = 1; i < Nx + 1; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            position_v_d[i][j] = position_v[i][j] - dt * eulerian_v_d[i][j];

            position_v_d[0][j]      = position_v[0][j];
            position_v_d[Nx + 1][j] = position_v[Nx + 1][j];
        }
    }
}

// 线性插值的基函数
double L0(double x) { return 1.0 - x; }
double L1(double x) { return x; }
double (*L[2])(double) = {L0, L1};

// 双线性插值
template <typename TV, int DIM = 2>
TV bilinear(double x, double y, TV* f) {
    TV sum = 0.0;
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            sum += f[j * DIM + i] * L[i](x) * L[j](y);
    return sum;
}

void trace(std::vector<std::vector<double2>>& eulerian_u_new, std::vector<std::vector<double2>>& eulerian_v_new,
           const std::vector<std::vector<double2>>& position_u, const std::vector<std::vector<double2>>& position_v,
           const std::vector<std::vector<double>>& eulerian_u, const std::vector<std::vector<double>>& eulerian_v,
           double dt, double hx, double hy, size_t Nx, size_t Ny) {
    for (size_t i = 0; i < Nx + 1; i++) {
        for (size_t j = 0; j < Ny + 2; j++) {
            double2 position = position_u[i][j];
            {
                // 计算出发点落在哪个单元格中
                size_t ii = position.x / hx;
                size_t jj = (position.y + 0.5 * hy) / hy;
                ii        = ii > Nx - 1 ? Nx - 1 : ii;
                jj        = jj > Ny ? Ny : jj;
                ii        = ii < 0 ? 0 : ii;
                jj        = jj < 0 ? 0 : jj;

                // 根据单元格索引取出数据
                double f[4];
                f[0] = eulerian_u[ii][jj];         // (0,0)
                f[1] = eulerian_u[ii + 1][jj];     // (1,0)
                f[2] = eulerian_u[ii][jj + 1];     // (0,1)
                f[3] = eulerian_u[ii + 1][jj + 1]; // (1,1)

                // 计算出发点在单元格中的局部坐标
                double local_x = (position.x - ii * hx) / hx;
                double local_y = (position.y + 0.5 * hy - jj * hy) / hy;

                // 双线性插值
                eulerian_u_new[i][j].x = bilinear<double, 2>(local_x, local_y, f);

                // LOG_F(INFO, "出发点坐标 %f %f %f", local_x, local_y,
                // eulerian_u_new[i][j].x); LOG_F(INFO, "出发点坐标 %f %f %f %f",
                // f[0],f[1],f[2],f[3]);
            }
            {
                // 计算出发点落在哪个单元格中
                size_t ii = (position.x + 0.5 * hx) / hx;
                size_t jj = position.y / hy;
                ii        = ii > Nx ? Nx : ii;
                jj        = jj > Ny - 1 ? Ny - 1 : jj;
                ii        = ii < 0 ? 0 : ii;
                jj        = jj < 0 ? 0 : jj;

                // 根据单元格索引取出数据
                double f[4];
                f[0] = eulerian_v[ii][jj];         // (0,0)
                f[1] = eulerian_v[ii + 1][jj];     // (1,0)
                f[2] = eulerian_v[ii][jj + 1];     // (0,1)
                f[3] = eulerian_v[ii + 1][jj + 1]; // (1,1)

                // 计算出发点在单元格中的局部坐标
                double local_x = (position.x + 0.5 * hx - ii * hx) / hx;
                double local_y = (position.y - jj * hy) / hy;

                // 双线性插值
                eulerian_u_new[i][j].y = bilinear<double, 2>(local_x, local_y, f);
            }
        }
    }
    for (size_t i = 0; i < Nx + 2; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            double2 position = position_v[i][j];
            {
                // 计算出发点落在哪个单元格中
                size_t ii = position.x / hx;
                size_t jj = (position.y + 0.5 * hy) / hy;
                ii        = ii > Nx - 1 ? Nx - 1 : ii;
                jj        = jj > Ny ? Ny : jj;
                ii        = ii < 0 ? 0 : ii;
                jj        = jj < 0 ? 0 : jj;

                // 根据单元格索引取出数据
                double f[4];
                f[0] = eulerian_u[ii][jj];         // (0,0)
                f[1] = eulerian_u[ii + 1][jj];     // (1,0)
                f[2] = eulerian_u[ii][jj + 1];     // (0,1)
                f[3] = eulerian_u[ii + 1][jj + 1]; // (1,1)

                // 计算出发点在单元格中的局部坐标
                double local_x = (position.x - ii * hx) / hx;
                double local_y = (position.y + 0.5 * hy - jj * hy) / hy;

                // 双线性插值
                eulerian_v_new[i][j].x = bilinear<double, 2>(local_x, local_y, f);

                // LOG_F(INFO, "出发点坐标 %f %f %f", local_x, local_y,
                // eulerian_u_new[i][j].x); LOG_F(INFO, "出发点坐标 %f %f %f %f",
                // f[0],f[1],f[2],f[3]);
            }
            {
                // 计算出发点落在哪个单元格中
                size_t ii = (position.x + 0.5 * hx) / hx;
                size_t jj = position.y / hy;
                ii        = ii > Nx ? Nx : ii;
                jj        = jj > Ny - 1 ? Ny - 1 : jj;
                ii        = ii < 0 ? 0 : ii;
                jj        = jj < 0 ? 0 : jj;

                // 根据单元格索引取出数据
                double f[4];
                f[0] = eulerian_v[ii][jj];         // (0,0)
                f[1] = eulerian_v[ii + 1][jj];     // (1,0)
                f[2] = eulerian_v[ii][jj + 1];     // (0,1)
                f[3] = eulerian_v[ii + 1][jj + 1]; // (1,1)

                // 计算出发点在单元格中的局部坐标
                double local_x = (position.x + 0.5 * hx - ii * hx) / hx;
                double local_y = (position.y - jj * hy) / hy;

                // 双线性插值
                eulerian_v_new[i][j].y = bilinear<double, 2>(local_x, local_y, f);
            }
        }
    }
}

int trace(const std::vector<std::vector<double>>& u_i, const std::vector<std::vector<double>>& v_i,
          std::vector<std::vector<double>>& u_d_u, std::vector<std::vector<double>>& v_d_v, double dt, double dx,
          double dy, size_t Nx, size_t Ny) {
    LOG_F(INFO, "dx %f dy %f", dx, dy);
    auto u_d     = make_vector_2D<double2>(Nx + 1, Ny + 2);
    auto v_d     = make_vector_2D<double2>(Nx + 2, Ny + 1);
    auto u_d_new = make_vector_2D<double2>(Nx + 1, Ny + 2);
    auto v_d_new = make_vector_2D<double2>(Nx + 2, Ny + 1);
    auto p_u_i   = make_vector_2D<double2>(Nx + 1, Ny + 2);
    auto p_v_i   = make_vector_2D<double2>(Nx + 2, Ny + 1);
    auto p_u_d   = make_vector_2D<double2>(Nx + 1, Ny + 2);
    auto p_v_d   = make_vector_2D<double2>(Nx + 2, Ny + 1);

    // 计算出发点的速度 p_u_d, p_v_d
    trace(u_d, v_d, u_i, v_i, dt, dx, dy, Nx, Ny);

    // 计算当前的位置 p_u_i, p_v_i
    current(p_u_i, p_v_i, dt, dx, dy, Nx, Ny);
    int iter      = 0;
    int max_inter = 1;

    // while(algebra::distance(v_d, v_d_new)*std::sqrt(dx*dy) > 1e-6)
    while (true) {
        // 计算出发点的位置 p_u_d, p_v_d
        originated(u_d, v_d, p_u_d, p_v_d, p_u_i, p_v_i, dt, dx, dy, Nx, Ny);

        // 计算出发点的速度 u_d, v_d
        trace(u_d_new, v_d_new, p_u_d, p_v_d, u_i, v_i, dt, dx, dy, Nx, Ny);

        // // 梯形求积公式
        // algebra::axpy(1.0, u_d, u_d_new);
        // algebra::axpy(1.0, v_d, v_d_new);
        // algebra::scale(0.5, u_d_new);
        // algebra::scale(0.5, v_d_new);

        double error_u = algebra::distance(u_d, u_d_new) * std::sqrt(dx * dy);
        double error_v = algebra::distance(v_d, v_d_new) * std::sqrt(dx * dy);

        std::cout << "半拉格朗日方法的收敛 u：" << error_u << std::endl;
        std::cout << "半拉格朗日方法的收敛 v：" << error_v << std::endl;

        ++iter;
        u_d = u_d_new;
        v_d = v_d_new;
        if (error_v < 1e-6) { break; }

        if (iter >= max_inter) { break; }
    }

    auto temp_1 = algebra::split(u_d);
    u_d_u       = std::get<0>(temp_1);
    auto temp_2 = algebra::split(v_d);
    v_d_v       = std::get<1>(temp_2);
    return 0;
}

} // namespace semi_lagrange
#endif