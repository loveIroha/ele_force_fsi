/// @date 2023-10-16
/// @file multiarray.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <AlgebraSolver/GridArray.h>
#include <PhysicsSolver/StokesFlow2D/MonolithicSolver.h>

#include <fstream>

void test_multiarray() {
    GridArray<double, 2> multi_array({4, 3});
    multi_array.fill_test();
    multi_array.print();
    multi_array.pad(1, 2, 3, 4);
    multi_array.print_raw();

    for (int j = -multi_array.padding.bottom; j < multi_array.size[1] + multi_array.padding.top; j++) {
        for (int i = -multi_array.padding.left; i < multi_array.size[0] + multi_array.padding.right; i++) {
            printf("array(%d, %d) =  %f\n", i, j, multi_array(i, j));
        }
    }
}

class SteadyStokesDirectSolver2D {
    double              width = 1.0, height = 1.0, T = 1.0;
    int                 Nx = 100, Ny = 100, Nt = 100;
    double              dt = T / Nt;
    double              hx = width / Nx, hy = height / Ny;
    double              mu  = 0.01;
    double              rho = 1;
    NavierStokesLocal2D nsl_2D;
    SteadyStokesDirectSolver2D() : nsl_2D(hx, hy, dt, mu, rho){};
};

int main() {
    double width = 1.0, height = 1.0, T = 1.0;
    int    Nx = 100, Ny = 100, Nt = 100;
    double dt = T / Nt;
    double hx = width / Nx, hy = height / Ny;
    double mu  = 0.01;
    double rho = 1;

    // 构造N-S方程的局部求解矩阵
    NavierStokesLocal2D nsl_2D(hx, hy, dt, mu, rho);

    std::cout << "矩阵：" << std::endl << nsl_2D.matrix << std::endl;

    std::cout << "逆矩阵：" << std::endl << nsl_2D.inverse << std::endl;

    // 创建变量
    GridArray<double, 2> f1({Nx, Ny});
    GridArray<double, 2> f2({Nx, Ny});

    GridArray<double, 2> un({Nx, Ny});
    GridArray<double, 2> vn({Nx, Ny});
    GridArray<double, 2> pn({Nx, Ny});
    GridArray<double, 2> sn({Nx, Ny});

    // for (size_t i = -1; i < Nx+1; i++)
    // {
    //     for (size_t j = -1; i < Ny+1; j++)
    //     {
    //         double x = (i+0.5)*hx ;
    //         double y = (j+0.5)*hy ;
    //         sn(i,j) = 2*x + y;
    //     }

    // }

    GridArray<double, 2> unn({Nx, Ny});
    GridArray<double, 2> vnn({Nx, Ny});
    GridArray<double, 2> pnn({Nx, Ny});

    GridArray<double, 2> ue({Nx, Ny});
    GridArray<double, 2> ve({Nx, Ny});
    GridArray<double, 2> pe({Nx, Ny});

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

    for (size_t it = 0; it < Nt; it++) {
        un.data = unn.data;
        vn.data = vnn.data;

        for (size_t n = 0; n < 20000; n++) {
            double error_sum = 0.0;
            // un.print_raw();

            // 设置边界条件
            for (int i = 1; i < Nx; i++) {
                unn(i, -1) = -unn(i, 0);
                unn(i, Ny) = 2.0 - unn(i, Ny - 1);
            }

            for (int j = -1; j < Ny + 1; j++) {
                unn(-1, j)     = -unn(1, j);
                unn(Nx + 1, j) = -unn(Nx - 1, j);
            }

            for (int j = 0; j < Ny + 1; j++) {
                vnn(j, -1)     = -vnn(j, 1);
                vnn(j, Ny + 1) = -vnn(j, Ny - 1);
            }

            for (int j = 0; j < Ny + 1; j++) {
                vnn(-1, j) = -vnn(0, j);
                vnn(Nx, j) = -vnn(Nx - 1, j);
            }

            for (int i = 0; i < Nx; i++) {
                pnn(i, -1) = pnn(i, 0);
                pnn(i, Ny) = pnn(i, Ny - 1);
            }

            for (int j = 0; j < Ny; j++) {
                pnn(-1, j) = pnn(0, j);
                pnn(Nx, j) = pnn(Nx - 1, j);
            }

            // un.print_raw();
            // vn.print_raw();
            // pn.print_raw();

            // 进行一次迭代
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    double b[5] = {0.0};
                    double x[5] = {0.0};
                    // b[0] = f1(i, j)                             - (un(i, j - 1)
                    // + un(i, j + 1))     / hy / hy - un(i - 1, j) / hx / hx -
                    // pn(i - 1, j) / hx; b[1] = f1(i + 1, j) - (un(i + 1, j - 1)
                    // + un(i + 1, j + 1)) / hy / hy -       un(i + 2, j) / hx /
                    // hx + pn(i + 1, j) / hx; b[2] = f2(i, j) - (vn(i - 1, j) +
                    // vn(i + 1, j))     / hx / hx - vn(i, j - 1) / hy / hy -
                    // pn(i, j - 1) / hy; b[3] = f2(i, j + 1) - (vn(i - 1, j
                    // + 1) + vn(i + 1, j + 1)) / hx / hx -       vn(i, j + 2) /
                    // hy / hy + pn(i, j + 1) / hy;
                    b[0] = f1(i, j) + pnn(i - 1, j) / hx + mu * (unn(i, j - 1) + unn(i, j + 1)) / hy / hy
                           + mu * unn(i - 1, j) / hx / hx;
                    +rho* un(i, j) / dt;
                    b[1] = f1(i + 1, j) - pnn(i + 1, j) / hx + mu * (unn(i + 1, j - 1) + unn(i + 1, j + 1)) / hy / hy
                           + mu * unn(i + 2, j) / hx / hx;
                    +rho* un(i + 1, j) / dt;
                    b[2] = f2(i, j) + pnn(i, j - 1) / hy + mu * (vnn(i - 1, j) + vnn(i + 1, j)) / hx / hx
                           + mu * vnn(i, j - 1) / hy / hy;
                    +rho* vn(i, j) / dt;
                    b[3] = f2(i, j + 1) - pnn(i, j + 1) / hy + mu * (vnn(i - 1, j + 1) + vnn(i + 1, j + 1)) / hx / hx
                           + mu * vnn(i, j + 2) / hy / hy;
                    +rho* vn(i, j + 1) / dt;
                    b[4] = sn(i, j);

                    nsl_2D.solve(x, b);
                    // printf("i, j = %d %d\n", i, j);
                    // printf("b = %f %f %f %f %f\n", b[0], b[1], b[2], b[3],
                    // b[4]); printf("x = %f %f %f %f %f\n", x[0], x[1], x[2],
                    // x[3], x[4]);

                    // 计算局部误差
                    double local_error = std::abs(unn(i, j) - x[0]) + std::abs(unn(i + 1, j) - x[1])
                                         + std::abs(vnn(i, j) - x[2]) + std::abs(vnn(i, j + 1) - x[3]);
                    +std::abs(pnn(i, j) - x[4]);
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
            printf("n = %d, error_sum = %.16e\n", n, error_sum);
            if (error_sum < 1e-6) { break; }
        }
    }

    std::ofstream out_r_prime("result.txt");

    un.print_raw(out_r_prime);
    vn.print_raw(out_r_prime);
    pn.print_raw(out_r_prime);
    ue.print_raw(out_r_prime);
    ve.print_raw(out_r_prime);
    pe.print_raw(out_r_prime);

    for (size_t j = 0; j < Ny; j++) {
        /* code */

        printf("%.16e\n", vn(5, j));
    }
}