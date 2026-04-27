/// @date 2023-10-18
/// @file boxsmoothing_2D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/StokesFlow2D/BoxSmoothing/BoxSmoothing.h>

int main() {
    boxsmoothing::StokesFlow<2> stokes_flow;

    for (int it = 0; it < stokes_flow.Nt; it++) {
        stokes_flow._t += stokes_flow.dt;
        // set s
        for (int ix = 0; ix < stokes_flow.Nx; ix++) {
            for (int iy = 0; iy < stokes_flow.Ny; iy++) {
                double x               = 0.5 * stokes_flow.hx + ix * stokes_flow.hx;
                double y               = 0.5 * stokes_flow.hy + iy * stokes_flow.hy;
                stokes_flow.sn(ix, iy) = -100 * (1 - x) * x * y * (1 - y);
            }
        }
        stokes_flow.solve_one_step();

        stokes_flow.un.data = stokes_flow.unn.data;
        stokes_flow.vn.data = stokes_flow.vnn.data;
        stokes_flow.pn.data = stokes_flow.pnn.data;

        stokes_flow.write_result();
    }

    for (int j = 0; j < stokes_flow.Ny; j++) {
        printf("%.16e\n", stokes_flow.vn(5, j));
    }
    return 0;
}
// int main_1()
// {
//     double width = 1.0, height = 1.0, T = 1.0;
//     int Nx = 10, Ny = 10, Nt = 1;
//     double dt = T / Nt;
//     double hx = width / Nx, hy = height / Ny;
//     double mu = 0.01;
//     double rho = 1;
//     VTIWriter writer("data/result.pvd");

//     // 构造N-S方程的局部求解矩阵
//     NavierStokesLocal2D nsl_2D(hx, hy, dt, mu, rho);

//     std::cout << "矩阵：" << std::endl
//               << nsl_2D.matrix << std::endl;

//     std::cout << "逆矩阵：" << std::endl
//               << nsl_2D.inverse << std::endl;

//     // 创建变量
//     GridArray<double, 2> f1({Nx, Ny});
//     GridArray<double, 2> f2({Nx, Ny});

//     GridArray<double, 2> un({Nx, Ny});
//     GridArray<double, 2> vn({Nx, Ny});
//     GridArray<double, 2> pn({Nx, Ny});
//     GridArray<double, 2> sn({Nx, Ny});

//     GridArray<double, 2> unn({Nx, Ny});
//     GridArray<double, 2> vnn({Nx, Ny});
//     GridArray<double, 2> pnn({Nx, Ny});

//     GridArray<double, 2> ue({Nx, Ny});
//     GridArray<double, 2> ve({Nx, Ny});
//     GridArray<double, 2> pe({Nx, Ny});

//     ue.pad(1, 2, 1, 1);
//     un.pad(1, 2, 1, 1);
//     unn.pad(1, 2, 1, 1);
//     f1.pad(1, 2, 1, 1);

//     ve.pad(1, 1, 1, 2);
//     vn.pad(1, 1, 1, 2);
//     vnn.pad(1, 1, 1, 2);
//     f2.pad(1, 1, 1, 2);

//     pe.pad(1, 1, 1, 1);
//     pn.pad(1, 1, 1, 1);
//     pnn.pad(1, 1, 1, 1);
//     sn.pad(1, 1, 1, 1);

//     for (int it = 0; it < Nt; it++)
//     {

//         // set f1
//         for (int ix = -1; ix < Nx + 1; ix++)
//         {
//             for (int iy = 0; iy < Ny + 1; iy++)
//             {
//                 /* code */
//             }
//         }
//         // set f2
//         // set s
//         for (int ix = 0; ix < Nx; ix++)
//         {
//             for (int iy = 0; iy < Ny; iy++)
//             {
//                 double x = 0.5 * hx + ix * hx;
//                 double y = 0.5 * hy + iy * hy;
//                 sn(ix, iy) = -100 * (1 - x) * x * y * (1 - y);
//             }
//         }

//         for (int n = 0; n < 20000; n++)
//         {
//             // 设置边界条件
//             boundary_conditions(unn, vnn, pnn);

//             // 进行一次迭代
//             double error_sum = solve_one_step(f1, f2, un, vn, pn, sn, unn,
//             vnn, pnn, ue, ve, pe, nsl_2D);

//             printf("n = %d, error_sum = %.16e\n", n, error_sum);
//             // 计算误差
//             if (error_sum < 1e-12)
//             {
//                 break;
//             }
//         }
//         un.data = unn.data;
//         vn.data = vnn.data;
//         pn.data = pnn.data;

//         GridArray<double2, 2> cu({Nx, Ny});
//         GridArray<double2, 2> cf({Nx, Ny});
//         GridArray<double, 2> cp({Nx, Ny});

//         cu.pad(1, 1, 1, 1);
//         cf.pad(1, 1, 1, 1);
//         cp.pad(1, 1, 1, 1);

//         for (int i = 0; i < Nx; i++)
//         {
//             for (int j = 0; j < Ny; j++)
//             {
//                 cu(i, j).x = 0.5 * (un(i, j), un(i + 1, j));
//                 cu(i, j).y = 0.5 * (vn(i, j) + vn(i, j + 1));
//                 cp(i, j) = pn(i, j);
//             }
//         }

//         writer.write(
//             make_int2(Nx + 2, Ny + 2),
//             make_double2(-0.5 * hx, -0.5 * hy),
//             make_double2(hx, hy),
//             cu.data,
//             cf.data,
//             pn.data, it);
//     }

//     for (int j = 0; j < Ny; j++)
//     {
//         printf("%.16e\n", vn(5, j));
//     }
//     return 0;
// }

// void test_multiarray()
// {
//     GridArray<double, 2> multi_array({4, 3});
//     multi_array.fill_test();
//     multi_array.print();
//     multi_array.pad(1, 2, 3, 4);
//     multi_array.print_raw();
//     for (int j = -multi_array.padding.bottom; j < multi_array.size[1] +
//     multi_array.padding.top; j++)
//     {
//         for (int i = -multi_array.padding.left; i < multi_array.size[0] +
//         multi_array.padding.right; i++)
//         {
//             printf("array(%d, %d) =  %f\n", i, j, multi_array(i, j));
//         }
//     }
// }

// int convert_GridArray_to_vector_f1(
//     const GridArray<double, 2> &f1,
//     const GridArray<double, 2> &un,
//     GridArray<double, 2> &unn,
//     GridArray<double, 2> &ue,
//     std::vector<std::vector<double>> &f1_ib,
//     std::vector<std::vector<double>> &un_ib,
//     std::vector<std::vector<double>> &unn_ib,
//     std::vector<std::vector<double>> &ue_ib)
// {
//     for (size_t i = -1; i < Nx + 1; i++)
//     {
//         for (size_t j = 0; i < Ny + 1; i++)
//         {
//             f1_ib[i + 1][j] = f1(i, j);
//             un_ib[i + 1][j] = un(i, j);
//             unn_ib[i + 1][j] = unn(i, j);
//             ue_ib[i + 1][j] = ue(i, j);
//         }
//     }
// }
// int convert_GridArray_to_vector_f2(
//     const GridArray<double, 2> &f2,
//     const GridArray<double, 2> &vn,
//     GridArray<double, 2> &vnn,
//     GridArray<double, 2> &ve,
//     std::vector<std::vector<double>> &f2_ib,
//     std::vector<std::vector<double>> &vn_ib,
//     std::vector<std::vector<double>> &vnn_ib,
//     std::vector<std::vector<double>> &ve_ib)
// {
//     for (size_t i = 0; i < Nx + 1; i++)
//     {
//         for (size_t j = -1; j < Ny + 1; j++)
//         {
//             f2_ib[i][j + 1] = f2(i, j);
//             vn_ib[i][j + 1] = vn(i, j);
//             vnn_ib[i][j + 1] = vnn(i, j);
//             ve_ib[i][j + 1] = ve(i, j);
//         }
//     }
// }
// int convert_vector_f1_to_GridArray(
//     const GridArray<double, 2> &f1,
//     const GridArray<double, 2> &un,
//     GridArray<double, 2> &unn,
//     GridArray<double, 2> &ue,
//     std::vector<std::vector<double>> &f1_ib,
//     std::vector<std::vector<double>> &un_ib,
//     std::vector<std::vector<double>> &unn_ib,
//     std::vector<std::vector<double>> &ue_ib)
// {
//     for (size_t i = -1; i <  Nx + 1; i++)
//     {
//         for (size_t j = 0; i < Ny + 1; i++)
//         {
//             f1(i, j) = f1_ib[i + 1][j];
//             un(i, j) = un_ib[i + 1][j];
//             unn(i, j) = unn_ib[i + 1][j];
//             ue(i, j) = ue_ib[i + 1][j];
//         }
//     }
// }
// int convert_vector_f2_to_GridArray(
//     const GridArray<double, 2> &f2,
//     const GridArray<double, 2> &vn,
//     GridArray<double, 2> &vnn,
//     GridArray<double, 2> &ve,
//     std::vector<std::vector<double>> &f2_ib,
//     std::vector<std::vector<double>> &vn_ib,
//     std::vector<std::vector<double>> &vnn_ib,
//     std::vector<std::vector<double>> &ve_ib)
// {
//     for (size_t i = 0; i < Nx + 1; i++)
//     {
//         for (size_t j = -1; j < Ny + 1; j++)
//         {
//             f2(i, j) = f2_ib[i][j + 1];
//             vn(i, j) = vn_ib[i][j + 1];
//             vnn(i, j) = vnn_ib[i][j + 1];
//             ve(i, j) = ve_ib[i][j + 1];
//         }
//     }
// }
// int solve_one_step(
//     const GridArray<double, 2> &f1,
//     const GridArray<double, 2> &f2,
//     const GridArray<double, 2> &un,
//     const GridArray<double, 2> &vn,
//     const GridArray<double, 2> &pn,
//     const GridArray<double, 2> &sn,
//     GridArray<double, 2> &unn,
//     GridArray<double, 2> &vnn,
//     GridArray<double, 2> &pnn,

//     GridArray<double, 2> &ue,
//     GridArray<double, 2> &ve,
//     GridArray<double, 2> &pe)
// {
//     auto u_ = unn;
//     auto v_ = vnn;

//     solve_one_step(
//         f1, f2, un, vn, pn, sn,
//         unn, vnn, pnn, ue, ve, pe);
//     return 0;
// }

// class SteadyStokesDirectSolver2D
// {
//     double width = 1.0, height = 1.0, T = 1.0;
//     int Nx = 10, Ny = 10, Nt = 1;
//     double dt = T / Nt;
//     double hx = width / Nx, hy = height / Ny;
//     double mu = 0.01;
//     double rho = 1;
//     NavierStokesLocal2D nsl_2D;
//     SteadyStokesDirectSolver2D() : nsl_2D(hx, hy, dt, mu, rho){};
// };

// int boundary_conditions(
//     GridArray<double, 2> &unn,
//     GridArray<double, 2> &vnn,
//     GridArray<double, 2> &pnn)
// {
//     int Nx = 10, Ny = 10;

//     // 上边界和下边界
//     unn(0, -1) = -unn(0, 0);
//     unn(0, Ny) = -unn(0, Ny - 1);
//     unn(Nx, -1) = -unn(Nx, 0);
//     unn(Nx, Ny) = -unn(Nx, Ny - 1);
//     for (int i = 1; i < Nx; i++)
//     {
//         unn(i, -1) = -unn(i, 0);
//         unn(i, Ny) = 2.0 * 1.0 - unn(i, Ny - 1);
//     }

//     // 左边界和右边界
//     for (int j = 0; j < Ny + 1; j++)
//     {
//         unn(-1, j) = -unn(1, j);
//         unn(Nx + 1, j) = -unn(Nx - 1, j);
//     }

//     for (int i = 0; i < Nx; i++)
//     {
//         vnn(i, -1) = vnn(i, 1);
//         vnn(i, Ny + 1) = -vnn(i, Ny - 1);
//     }

//     for (int j = 0; j < Ny + 1; j++)
//     {
//         vnn(-1, j) = -vnn(0, j);
//         vnn(Nx, j) = -vnn(Nx - 1, j);
//     }

//     for (int i = 0; i < Nx; i++)
//     {
//         pnn(i, -1) = pnn(i, 0);
//         pnn(i, Ny) = -pnn(i, Ny - 1);
//     }

//     for (int j = 0; j < Ny; j++)
//     {
//         pnn(-1, j) = pnn(0, j);
//         pnn(Nx, j) = pnn(Nx - 1, j);
//     }
//     return 0;
// }
