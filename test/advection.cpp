/// @date 2023-06-26
/// @file advection.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试半拉格朗日方法
///
///

#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven2D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow2D/StokesFlow.h>
#include <dolfin.h>
#include <io/writeVTK.h>

#include <tuple>

const int degree = 1;
const int DIM    = 2;

using namespace semi_lagrange;

double2 function_velocity(const double3& x) {
    double a = x.x * x.x + x.y * x.y;
    double b = x.y * x.y;
    return make_double2(a, b);
}

int main() {
    size_t Nx = 32, Ny = 32;

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(10,         // Nt
                                    {Nx, Ny},   // Nx Ny
                                    {1.0, 1.0}, // Lx Ly
                                    0.01,       // T
                                    1.0,        // rho
                                    0.01        // mu
    );

    double dt = domain_mesh._dt;
    double dx = domain_mesh._dx;
    double dy = domain_mesh._dy;

    auto u_i = make_vector_2D<double>(Nx + 1, Ny + 2);
    auto v_i = make_vector_2D<double>(Nx + 2, Ny + 1);
    auto u_d = make_vector_2D<double>(Nx + 1, Ny + 2);
    auto v_d = make_vector_2D<double>(Nx + 2, Ny + 1);

    // 任意给个流场速度
    auto [u_i_1D, v_i_1D] = domain_mesh.set_function_on_staggered_grid(function_velocity);
    u_i                   = algebra::ripple(u_i_1D, u_i.size());
    v_i                   = algebra::ripple(v_i_1D, v_i.size());

    trace(u_i, v_i, u_d, v_d, dt, dx, dy, Nx, Ny);

    // 输出延拓算子作用后的结果
    std::string filename_u  = "results/function_u.vti";
    std::string arrayname_u = "function_u";
    write_vtk(domain_mesh.get_dim_u(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
              make_double2(domain_mesh._dx, domain_mesh._dy), u_i_1D, filename_u, arrayname_u);

    std::string filename_v  = "results/function_v.vti";
    std::string arrayname_v = "function_v";
    write_vtk(domain_mesh.get_dim_v(), make_double2(domain_mesh.origin.x, domain_mesh.origin.y),
              make_double2(domain_mesh._dx, domain_mesh._dy), v_i_1D, filename_v, arrayname_v);

    return 0;
}
