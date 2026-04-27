/// @date 2024-01-11
/// @file advection_only.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief  检查线性对流算法
///
///

#include <advection/LinearAdvection.h>
#include <io.h>

int test_linear_advection() {
    unsigned int       n = 96;
    auto advection = std::make_unique<LinearAdvection>(n, n, n, 1.0/n, 1.0, 1.0, 1.0);

    // std::vector<double> cpu_u((n + 3) * (n + 2) * (n + 2));
    // std::vector<double> cpu_v((n + 2) * (n + 3) * (n + 2));
    // std::vector<double> cpu_w((n + 2) * (n + 2) * (n + 3));

    auto cpu_u = IO::make_vector_3D<double>(n + 3,n + 2, n + 2);
    auto cpu_v = IO::make_vector_3D<double>(n + 2,n + 3, n + 2);
    auto cpu_w = IO::make_vector_3D<double>(n + 2,n + 2, n + 3);



        for (int z = 0; z < n + 2; z++) {
            for (int y = 0; y < n + 2; y++) {
                for (int x = 0; x < n + 3; x++) {
                    // cpu_u[x][y][z] = x - 1.0;
                    // cpu_u[x][y][z] = y - 0.5;
                    cpu_u[x][y][z] = z - 0.5;
                }
            }
        }
        for (int z = 0; z < n + 2; z++) {
            for (int y = 0; y < n + 3; y++) {
                for (int x = 0; x < n + 2; x++) {
                    // cpu_v[x][y][z] = x - 0.5;
                    // cpu_v[x][y][z] = y - 1.0;
                    cpu_v[x][y][z] = z - 0.5;
                }
            }
        }
        for (int z = 0; z < n + 3; z++) {
            for (int y = 0; y < n + 2; y++) {
                for (int x = 0; x < n + 2; x++) {
                    // cpu_w[x][y][z] = x - 0.5;
                    // cpu_w[x][y][z] = y - 0.5;
                    cpu_w[x][y][z] = z - 1.0;
                }
            }
        }
    int i = 2, j =3, k = 4;
    LOG_F(INFO, "data u: %d %d %d %f", i, j, k, cpu_u[i][j][k]);
    LOG_F(INFO, "data v: %d %d %d %f", i, j, k, cpu_v[i][j][k]);
    LOG_F(INFO, "data w: %d %d %d %f", i, j, k, cpu_w[i][j][k]);

    for (int frame = 1; frame < 2000000; frame++) {
        // if (frame % 100 == 0) LOG_F(INFO, "frame=%d", frame);
        LOG_F(WARNING, "计算对流项");
        ScopeProfiler _{"kokkos::StokesFlow::solve_one_step::advection"};
        advection->advection(cpu_u, cpu_v, cpu_w);
    }

    LOG_F(INFO, "data u: %d %d %d %f", i, j, k, cpu_u[i][j][k]);
    LOG_F(INFO, "data v: %d %d %d %f", i, j, k, cpu_v[i][j][k]);
    LOG_F(INFO, "data w: %d %d %d %f", i, j, k, cpu_w[i][j][k]);

//   CHECK(Adense(4, 4) != Aref(4, 4));

    return 0;
}

int main() { 
for (size_t i = 0; i < 10000; i++)
    {
        test_linear_advection(); 
    }
}