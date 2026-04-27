/// @date 2023-09-28
/// @file NavierStokesDemo.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 从Json文件读入三维NS方程解析解，并生成可计算对象。
///
///

#include <PhysicsSolver/StokesFlow3D/LidDriven.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo_deprecated.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <advection.h>
#include <fmt/core.h>
#include <io/ScopeProfiler.h>
#include <io/writeVTK.h>
#include <mpParser.h>
#include <muParser.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <variant>

void fun(int Ns, int Nt) {
    test_linear_advection();
    [[maybe_unused]] double dt = 0.001;
    double                  T  = 0.01;
    // int Nt = std::ceil(T / dt);
    // int Nt = 4;

    int Nx = Ns;
    int Ny = Ns;
    int Nz = Ns;

    double Lx = 1.0;
    double Ly = 1.0;
    double Lz = 1.0;

    double rho = 1.0;
    double mu  = 1.0;

    stokes_flow::LidDriven<3> sf(Nt, {Nx, Ny, Nz}, {Lx, Ly, Lz}, T, rho, mu);

    sf.main_helmholtz();

    //
    auto velocity_center = algebra::create_multi_array<3, double3>({Nx + 2, Ny + 2, Nz + 2});
    stagger_to_center<3, double, double3>(velocity_center, sf.un, sf.vn, sf.wn, {Nx, Ny, Nz});

    // 输出速度在边界上的值
    std::ofstream out_ubv("ubv.txt");
    std::ofstream out_vbv("vbv.txt");
    std::ofstream out_wbv("wbv.txt");
    algebra::print_multi_array<3, double>(sf.ubv, out_ubv);
    algebra::print_multi_array<3, double>(sf.vbv, out_vbv);
    algebra::print_multi_array<3, double>(sf.wbv, out_wbv);

    // 输出原始的速度分量u
    int3        dim       = {Nx + 1, Ny + 2, Nz + 2};
    double3     origin    = {0.0, 0.0, 0.0};
    double3     dh        = {Lx / Nx, Ly / Ny, Lz / Nz};
    std::string filename  = "data/test.vti";
    std::string arrayname = "test";
    write_vtk(dim, origin, dh, algebra::flatten(sf.un), filename, arrayname);
    std::ofstream out_v_u("velocity_u.txt");
    algebra::print_multi_array<3, double>(sf.un, out_v_u);

    // 输出速度在中心点上的值
    int3        dim_center      = {Nx + 2, Ny + 2, Nz + 2};
    std::string filename_center = "data/test_center.vti";
    write_vtk(dim_center, origin, dh, algebra::flatten(velocity_center), filename_center, arrayname);
    std::ofstream out_vc("velocity_center.txt");
    algebra::print_multi_array<3, double3>(velocity_center, out_vc);

    // 从中心点拆分出速度分量u
    auto un = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
    center_to_stagger<3, double, double3>(un, sf.vn, sf.wn, velocity_center, {Nx, Ny, Nz});
    filename  = "data/test_u_new.vti";
    arrayname = "test_u_new";
    write_vtk(dim, origin, dh, algebra::flatten(un), filename, arrayname);
    std::ofstream out_v_u_new("velocity_v_u_new.txt");
    algebra::print_multi_array<3, double>(un, out_v_u_new);

    auto error = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
    algebra::axpy(-1.0, sf.un, un, error);
    std::ofstream out_v_u_e("velocity_v_u_error.txt");
    algebra::zero_boundary(error);

    algebra::print_multi_array<3, double>(error, out_v_u_e);
    printf("Error at the last step: %.16e\n", algebra::norm(error) * std::sqrt(dh.x * dh.y * dh.z));
}

int main() {
    std::vector<int> Nt = {16};
    std::vector<int> Ns = {32};
    for (const auto& ns : Ns) {
        for (const auto& nt : Nt) {
            fun(ns, nt);
        }
    }
    printScopeProfiler();
}
