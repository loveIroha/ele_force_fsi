/// @date 2023-12-26
/// @file algebra_kokkos.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#pragma once

#include <AlgebraSolver/algebra.h>
#include <Kokkos_Core.hpp>

template <typename T>
using View3D = Kokkos::View<T***>;

template <typename T>
using View3D_h = Kokkos::View<T***>::HostMirror;
namespace algebra {

template <typename L>
double norm(const View3D<L>& data, Norm norm_type = Norm::l2) {
    CHECK_F(norm_type == Norm::l2, "Only implemented for l2 norm.");

    auto Nx = data.extent(0);
    auto Ny = data.extent(1);
    auto Nz = data.extent(2);

    L result{};

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_reduce(
        "restrict", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& result) {
            result += data(i, j, k) * data(i, j, k);
        },
        result);
    return std::sqrt(result);
}

template <typename L>
void add(View3D<L>& a, const View3D<L>& b) {
    auto Nx = a.extent(0);
    auto Ny = a.extent(1);
    auto Nz = a.extent(2);

    assert(a.extent(0) == b.extent(0) && a.extent(1) == b.extent(1) && a.extent(2) == b.extent(2));

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
        "template <typename L> void add(View3D<L>& a, const View3D<L>& b);", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { a(i, j, k) += b(i, j, k); });
}

template <typename L>
[[deprecated("只允许在多重网格方法内部使用")]] double sum(const View3D<L>& data) {
    auto Nx = data.extent(0);
    auto Ny = data.extent(1);
    auto Nz = data.extent(2);
    L    result{};

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_reduce(
        "restrict", mdrange_policy({1, 1, 1}, {Nx - 1, Ny - 1, Nz - 1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& result) { result += data(i, j, k); }, result);
    return result / ((Nx - 2) * (Ny - 2) * (Nz - 2));
}

template <typename L>
[[deprecated("只允许在多重网格方法内部使用")]] void add(View3D<L>& a, L b) {
    auto Nx = a.extent(0);
    auto Ny = a.extent(1);
    auto Nz = a.extent(2);

    assert(a.extent(0) == b.extent(0) && a.extent(1) == b.extent(1) && a.extent(2) == b.extent(2));

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
        "template <typename L> void add(View3D<L>& a, L b)", mdrange_policy({1, 1, 1}, {Nx - 1, Ny - 1, Nz - 1}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { a(i, j, k) += b; });
}

template <typename L>
void zero(View3D<L>& a) {
    auto Nx = a.extent(0);
    auto Ny = a.extent(1);
    auto Nz = a.extent(2);

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
        "template <typename L> void zero(View3D<L>& a);", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { a(i, j, k) = L{}; });
}
} // namespace algebra