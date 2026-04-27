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
    /// 指定保存计算结果的文件
    std::string output_file = "lid_driven/velocity/data.pvd";
    VTIWriter   velocity_writer{output_file};

    BackgroundMesh2D<2> domain_mesh(Nt,       // Nt
                                    {Nx, Ny}, // Nx Ny
                                    {Lx, Ly}, // Lx Ly
                                    T,        // T
                                    1.0,      // rho
                                    0.01      // mu
    );

    StokesFlow<DIM> stokes_flow(Nt, {Nx, Ny}, {Lx, Ly}, T, 1.0, 0.01);

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

        // 输出某个点处的值
        // int i = _Nx / 2;
        // int i = 10;
        // int i = _Nx;
        std::cout << std::endl;
        std::cout << std::endl;
        for (size_t i = 0; i < _Nx + 1; i = i + 1) {
            for (size_t j = 0; j < _Ny + 2; j = j + 1) {
                // LOG_F(INFO, "u component of velocity at point %f %f is %f", i *
                // stokes_flow._dx, j * stokes_flow._dy, (u[i][j] + u[i][j + 1]) /
                // 2); LOG_F(INFO, "u component of velocity at point %f %f is %f",
                // i * stokes_flow._dx, j * stokes_flow._dy, u[i][j]);
                // printf("%3.3e    ", u[i][j]);
            }
            // std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < _Nx + 2; i = i + 1) {
            for (size_t j = 0; j < _Ny + 2; j = j + 1) {
                // LOG_F(INFO, "u component of velocity at point %f %f is %f", i *
                // stokes_flow._dx, j * stokes_flow._dy, (u[i][j] + u[i][j + 1]) /
                // 2); LOG_F(INFO, "u component of velocity at point %f %f is %f",
                // i * stokes_flow._dx, j * stokes_flow._dy, u[i][j]);
                // printf("%3.3e    ", p[i][j]);
            }
            // std::cout << std::endl;
        }

        auto cell_center_velocity = domain_mesh.cell_center_velocity(un, vn);
        if (step % 10 == 0)
            velocity_writer.write(domain_mesh.get_dim(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
                                  make_double2(domain_mesh._dx, domain_mesh._dy),
                                  algebra::flatten(cell_center_velocity), t);

        // for (size_t j = 0; j < _Ny + 2; j = j + 1)
        //     LOG_F(INFO, "Velocity at point %f %f is %f %f",
        //           (_Nx / 2) * stokes_flow._dx, j * stokes_flow._dy,
        //           cell_center_velocity[_Nx / 2][j].x, cell_center_velocity[_Nx
        //           / 2][j].y);

        if (t > stokes_flow._T - EPSILON) break;
    }
}

int main() {
    // int Nx = 220;
    // int Ny = 41;
    double width  = 2.0;
    double height = 2.0;
    int    Nx     = 128;
    int    Ny     = 128;
    int    Nt     = 200;
    double T      = 2;
    test_lid_driven(Nt, T, width, height, Nx, Ny);
}
