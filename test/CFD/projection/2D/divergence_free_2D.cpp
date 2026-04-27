/// @date 2023-06-14
/// @file mesh.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试流体求解器
///
///

#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven2D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <dolfin.h>
#include <io/writeVTK.h>

#include <tuple>

const int DIM = 2;

int test_lid_driven(int Nt, double T, double Lx, double Ly, int Nx, int Ny) {
    // 输入输出
    VTIWriter velocity_writer{"lid_driven/velocity/data.pvd"};

    BackgroundMesh2D<2> domain_mesh(Nt,       // Nt
                                    {Nx, Ny}, // Nx Ny
                                    {Lx, Ly}, // Lx Ly
                                    T,        // T
                                    1.0,      // rho
                                    0.001     // mu
    );

    StokesFlow<DIM> stokes_flow(Nt, {Nx, Ny}, {Lx, Ly}, T, 1.0, 0.001);

    int _Nx = stokes_flow._Nx;
    int _Ny = stokes_flow._Ny;

    auto u = make_vector_2D<double>(_Nx + 1, _Ny + 2);
    auto v = make_vector_2D<double>(_Nx + 2, _Ny + 1);
    auto p = make_vector_2D<double>(_Nx + 2, _Ny + 2);

    auto un = make_vector_2D<double>(_Nx + 1, _Ny + 2);
    auto vn = make_vector_2D<double>(_Nx + 2, _Ny + 1);

    auto uf = make_vector_2D<double>(_Nx + 1, _Ny + 2);
    auto vf = make_vector_2D<double>(_Nx + 2, _Ny + 1);

    // 设置边界类型
    int ubt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
    int vbt_edge[4] = {DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET};
    int pbt_edge[4] = {NEUMANN, NEUMANN, NEUMANN, NEUMANN};

    double velocity_u[4] = {0, 1, 0, 0};
    double velocity_v[4] = {0, 0, 0, 0};
    double pressure[4]   = {0, 0, 0, 0};

    stokes_flow.compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, velocity_u, velocity_v, pressure);

    // 给边界条件赋值
    double t  = 0.0;
    double dt = stokes_flow._dt;
    for (size_t step = 1; step < Nt; step++) {
        t = step * dt;

        // 计算出 u 和 v
        stokes_flow.solve_one_step(u, v, p, uf, vf, un, vn);

        // 将 u, v 赋值给 un, vn
        un = u;
        vn = v;

        for (size_t i = 0; i < _Nx; i++) {
            for (size_t j = 0; j < _Ny; j++) {
                double a = (u[i + 1][j + 1] - u[i][j + 1]) / stokes_flow._dx;
                a += (v[i + 1][j + 1] - v[i + 1][j]) / stokes_flow._dy;
                printf("a: %f\n", a);
            }
        }

        auto cell_center_velocity = domain_mesh.cell_center_velocity(un, vn);

        if (t > stokes_flow._T - EPSILON) break;
    }
}

int main() {
    // int Nx = 220;
    // int Ny = 41;
    double width  = 1.0;
    double height = 1.0;
    int    Nx     = 128;
    int    Ny     = 128;
    int    Nt     = 200;
    double T      = 1.0;
    test_lid_driven(Nt, T, width, height, Nx, Ny);
}
