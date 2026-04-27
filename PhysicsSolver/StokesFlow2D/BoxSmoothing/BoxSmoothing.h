/// @date 2023-11-16
/// @file BoxSmoothing.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include <AlgebraSolver/GridArray.h>
#include <PhysicsSolver/StokesFlow2D/BoxSmoothing/MonolithicSolver.h>
#include <io/vector_io.h>
#include <io/writeVTK.h>
#include <vector_functions.hpp>
#include <vector_types.h>

#include <fstream>

namespace boxsmoothing {
template <int DIM, class enable = void>
class StokesFlow;

template <int DIM>
class StokesFlow<DIM, std::enable_if_t<DIM == 2>> {
  public:
    int    Nx = 10, Ny = 10, Nt = 1, max_iters = 20000;
    double width = 1.0, height = 1.0, T = 1.0;
    double hx = width / Nx, hy = height / Ny, dt = T / Nt;
    double mu  = 0.01;
    double rho = 1;
    double _t  = 0.0;

    GridArray<double, 2> f1, f2;
    GridArray<double, 2> ue, ve, pe;
    GridArray<double, 2> unn, vnn, pnn;
    GridArray<double, 2> un, vn, pn, sn;

    // 构造N-S方程的局部求解矩阵
    VTIWriter           writer;
    NavierStokesLocal2D nsl_2D;

    int boundary_conditions(GridArray<double, 2>& unn, GridArray<double, 2>& vnn, GridArray<double, 2>& pnn) const {
        // const int Nx = 10, Ny = 10;

        // 上边界和下边界
        unn(0, -1)  = -unn(0, 0);
        unn(0, Ny)  = -unn(0, Ny - 1);
        unn(Nx, -1) = -unn(Nx, 0);
        unn(Nx, Ny) = -unn(Nx, Ny - 1);
        for (int i = 1; i < Nx; i++) {
            unn(i, -1) = -unn(i, 0);
            unn(i, Ny) = 2.0 * 1.0 - unn(i, Ny - 1);
        }

        // 左边界和右边界
        for (int j = 0; j < Ny + 1; j++) {
            unn(-1, j)     = -unn(1, j);
            unn(Nx + 1, j) = -unn(Nx - 1, j);
        }

        for (int i = 0; i < Nx; i++) {
            vnn(i, -1)     = vnn(i, 1);
            vnn(i, Ny + 1) = -vnn(i, Ny - 1);
        }

        for (int j = 0; j < Ny + 1; j++) {
            vnn(-1, j) = -vnn(0, j);
            vnn(Nx, j) = -vnn(Nx - 1, j);
        }

        for (int i = 0; i < Nx; i++) {
            pnn(i, -1) = pnn(i, 0);
            pnn(i, Ny) = -pnn(i, Ny - 1);
        }

        for (int j = 0; j < Ny; j++) {
            pnn(-1, j) = pnn(0, j);
            pnn(Nx, j) = pnn(Nx - 1, j);
        }
        return 0;
    }

    auto extract_u() const {
        auto u = IO::make_vector_2D<int>(Nx + 1, Ny + 2);
        for (size_t i = 0; i < Nx + 1; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                u[i][j] = un(i, j - 1);
            }
        }
        return u;
    }

    // 无状态函数
    double solve_one_step(const GridArray<double, 2>& f1, const GridArray<double, 2>& f2,
                          const GridArray<double, 2>& un, const GridArray<double, 2>& vn,
                          const GridArray<double, 2>& pn, const GridArray<double, 2>& sn, GridArray<double, 2>& unn,
                          GridArray<double, 2>& vnn, GridArray<double, 2>& pnn, GridArray<double, 2>& ue,
                          GridArray<double, 2>& ve, GridArray<double, 2>& pe) {
        double error_sum = 0.0;

        // 开始一次迭代
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                double b[5] = {0.0};
                double x[5] = {0.0};
                b[0]        = f1(i, j) + pnn(i - 1, j) / hx + mu * (unn(i, j - 1) + unn(i, j + 1)) / hy / hy
                       + mu * unn(i - 1, j) / hx / hx + rho * un(i, j) / dt;
                b[1] = f1(i + 1, j) - pnn(i + 1, j) / hx + mu * (unn(i + 1, j - 1) + unn(i + 1, j + 1)) / hy / hy
                       + mu * unn(i + 2, j) / hx / hx + rho * un(i + 1, j) / dt;
                b[2] = f2(i, j) + pnn(i, j - 1) / hy + mu * (vnn(i - 1, j) + vnn(i + 1, j)) / hx / hx
                       + mu * vnn(i, j - 1) / hy / hy + rho * vn(i, j) / dt;
                b[3] = f2(i, j + 1) - pnn(i, j + 1) / hy + mu * (vnn(i - 1, j + 1) + vnn(i + 1, j + 1)) / hx / hx
                       + mu * vnn(i, j + 2) / hy / hy + rho * vn(i, j + 1) / dt;
                b[4] = -sn(i, j);

                // 求解局部方程
                nsl_2D.solve(x, b);

                // 计算局部误差
                double local_error = std::abs(unn(i, j) - x[0]) + std::abs(unn(i + 1, j) - x[1])
                                     + std::abs(vnn(i, j) - x[2]) + std::abs(vnn(i, j + 1) - x[3])
                                     + std::abs(pnn(i, j) - x[4]);

                // printf("%d %d %.16e\n", i, j, local_error);
                error_sum += local_error;
                // printf("local_error = %f\n", local_error);

                ue(i, j)     = unn(i, j) - x[0];
                ue(i + 1, j) = unn(i + 1, j) - x[1];
                ve(i, j)     = vnn(i, j) - x[2];
                ve(i, j + 1) = vnn(i, j + 1) - x[3];
                pe(i, j)     = pnn(i, j) - x[4];

                // 提取出解
                unn(i, j)     = x[0];
                unn(i + 1, j) = x[1];
                vnn(i, j)     = x[2];
                vnn(i, j + 1) = x[3];
                pnn(i, j)     = x[4];
            }
        }
        return error_sum;
    }

    void solve_one_step() {
        for (int n = 0; n < max_iters; n++) {
            // 设置边界条件
            boundary_conditions(unn, vnn, pnn);

            // 进行一次迭代
            double error_sum = solve_one_step(f1, f2, un, vn, pn, sn, unn, vnn, pnn, ue, ve, pe);

            printf("n = %d, error_sum = %.16e\n", n, error_sum);
            // 计算误差
            if (error_sum < 1e-12) { break; }
        }
    }

    void solve_one_step(std::vector<std::vector<double>>& unn_, std::vector<std::vector<double>>& vnn_,
                        std::vector<std::vector<double>>& pnn_, const std::vector<std::vector<double>>& f1_,
                        const std::vector<std::vector<double>>& f2_, const std::vector<std::vector<double>>& un_,
                        const std::vector<std::vector<double>>& vn_, const std::vector<std::vector<double>>& sn_) {
        for (size_t i = 0; i < Nx + 1; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                f1(i, j - 1)  = f1_[i][j];
                un(i, j - 1)  = un_[i][j];
                unn(i, j - 1) = unn_[i][j];
            }
        }
        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 1; j++) {
                f2(i - 1, j)  = f2_[i][j];
                vn(i - 1, j)  = vn_[i][j];
                vnn(i - 1, j) = vnn_[i][j];
            }
        }
        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                sn(i - 1, j - 1)  = sn_[i][j];
                pnn(i - 1, j - 1) = pnn_[i][j];
            }
        }

        solve_one_step();

        for (size_t i = 0; i < Nx + 1; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                unn_[i][j] = unn(i, j - 1);
            }
        }
        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 1; j++) {
                vnn_[i][j] = vnn(i - 1, j);
            }
        }
        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 2; j++) {
                pnn_[i][j] = pnn(i - 1, j - 1);
            }
        }
    }

    void write_result() {
        // 输出结果
        GridArray<double2, 2> cu({Nx, Ny});
        GridArray<double2, 2> cf({Nx, Ny});
        GridArray<double, 2>  cp({Nx, Ny});

        cu.pad(1, 1, 1, 1);
        cf.pad(1, 1, 1, 1);
        cp.pad(1, 1, 1, 1);

        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                cu(i, j).x = 0.5 * (un(i, j), un(i + 1, j));
                cu(i, j).y = 0.5 * (vn(i, j) + vn(i, j + 1));
                cp(i, j)   = pn(i, j);
            }
        }

        writer.write(make_int2(Nx + 2, Ny + 2), make_double2(-0.5 * hx, -0.5 * hy), make_double2(hx, hy), cu.data,
                     cf.data, pn.data, _t);
    }

    StokesFlow(const std::string& path = "data/result.pvd")
        : f1({Nx, Ny}), f2({Nx, Ny}), ue({Nx, Ny}), ve({Nx, Ny}), pe({Nx, Ny}), unn({Nx, Ny}), vnn({Nx, Ny}),
          pnn({Nx, Ny}), un({Nx, Ny}), vn({Nx, Ny}), pn({Nx, Ny}), sn({Nx, Ny}), writer(path),
          nsl_2D(hx, hy, dt, mu, rho) {
        ue.pad(1, 2, 1, 1);
        un.pad(1, 2, 1, 1);
        unn.pad(1, 2, 1, 1);
        f1.pad(1, 2, 1, 1);

        ve.pad(1, 1, 1, 2);
        vn.pad(1, 1, 1, 2);
        vnn.pad(1, 1, 1, 2);
        f2.pad(1, 1, 1, 2);

        pe.pad(1, 1, 1, 1);
        pn.pad(1, 1, 1, 1);
        pnn.pad(1, 1, 1, 1);
        sn.pad(1, 1, 1, 1);
    }
};
} // namespace boxsmoothing