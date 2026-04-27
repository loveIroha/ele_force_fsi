/// @date 2023-12-28
/// @file some_headers.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#pragma once

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra_kokkos.h>
#include <config.h>
#include <io.h>
#include <sys/sysinfo.h>

template <typename T>
using Array3D  = algebra::Array3D<T>;
using Array3Di = algebra::Array3Di;
using Array3Dd = algebra::Array3Dd;

enum class Devices {
    CPU,
    GPU
};

namespace mykokkos {

template <Devices device>
void print_free_memory();

template <>
void print_free_memory<Devices::GPU>() {
    int            device;
    size_t         free, total;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaMemGetInfo(&free, &total);

    std::cout << "Total GPU Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Free GPU Memory: " << free / (1024 * 1024) << " MB\n";
}

template <>
void print_free_memory<Devices::CPU>() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) { CHECK_F(false, "Failed to get system information"); }
    std::cout << "Total RAM: " << info.totalram / static_cast<double>(1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << "Free RAM: " << info.freeram / static_cast<double>(1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << "Total swap: " << info.totalswap / static_cast<double>(1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << "Free swap: " << info.freeswap / static_cast<double>(1024 * 1024 * 1024) << " GB" << std::endl;
}

template <typename L>
auto to_View(const Array3D<L>& data_, std::string name = "data") {
    CHECK_F(data_.size() > 0, "data_.size() > 0");
    CHECK_F(data_[0].size() > 0, "data_[0].size() > 0");
    CHECK_F(data_[0][0].size() > 0, "data_[0][0].size() > 0");

    int Nx = data_.size();
    int Ny = data_[0].size();
    int Nz = data_[0][0].size();
    LOG_F(9, "Nx = %d, Ny = %d, Nz = %d", Nx, Ny, Nz);
    View3D<L> data(name, Nx, Ny, Nz);

    View3D_h<L> data_h = Kokkos::create_mirror_view(data);
    for (size_t i = 0; i < Nx; i++)
        for (size_t j = 0; j < Ny; j++)
            for (size_t k = 0; k < Nz; k++) {
                data_h(i, j, k) = data_[i][j][k];
            }
    Kokkos::deep_copy(data, data_h);

    return data;
}

template <typename KokkosSolver>
void solve(const std::unique_ptr<KokkosSolver>& kokkos_solver, Array3D<double>& rh, Array3D<double>& xh,
           const Array3D<int>& xbt, const Array3D<double>& xbv, const Array3D<double>& xb, int3 N, double3 L, double mu,
           double rho, double dt, int max_iters, double tolerance) {
    LOG_SCOPE_FUNCTION(WATCH);
    LOG_F(INFO, "高斯塞德尔迭代");
    LOG_F(WATCH, "max_iters = %d, tolerance  = %.20e", max_iters, tolerance);

    double3 dh = {L.x / N.x, L.y / N.y, L.z / N.z};
    kokkos_solver->load(xh, xb, xbv, xbt);
    kokkos_solver->generate_x_prime();
    kokkos_solver->generate_r_prime();
    for (int iter = 0; iter < max_iters; iter++) {
        kokkos_solver->smooth_boundary();
        kokkos_solver->smooth_inner(kokkos_solver->xh, kokkos_solver->xbct, kokkos_solver->r_prime, N, L);

        if (iter % 1 == 0) {
            kokkos_solver->smooth_boundary();
            kokkos_solver->residual();

            auto sum_r_squared = algebra::norm(kokkos_solver->xr) * std::sqrt(dh.x * dh.y * dh.z);
            LOG_F(WATCH, "iter = %d, sum_r_squared  = %.20e", iter, sum_r_squared);
            if (sum_r_squared < tolerance) { break; }
        }
    }
    algebra::add(kokkos_solver->xh, kokkos_solver->x_prime);
    kokkos_solver->extract(xh, rh);
}

} // namespace mykokkos