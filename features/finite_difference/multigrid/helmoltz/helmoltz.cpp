#include <iostream>
#include <vector>

#include "algebra.h"
#include "mglib.h"

namespace mg {
namespace u {
template <typename T>
vector2D<T> restrict(const vector2D<T>& fine, int Nx, int Ny) {
    vector2D<T> coarse = make_vector_2D<T>(Nx + 1, Ny + 2);
    for (int i = 0; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            coarse[i][j] = 0.5 * (fine[2 * i][2 * j] + fine[2 * i][2 * j - 1]);
        }
    }
    return coarse;
}

template <typename T>
vector2D<T> restrict_bc(const vector2D<T>& fine, int Nx, int Ny) {
    vector2D<T> coarse = make_vector_2D<T>(Nx + 1, Ny + 2);
    for (int i = 0; i <= Nx; ++i) // 上下边界
    {
        coarse[i][0]      = fine[2 * i][0];
        coarse[i][Ny + 1] = fine[2 * i][2 * Ny + 1];
    }
    for (int j = 1; j <= Ny; ++j) {
        coarse[0][j]  = std::max(fine[0][2 * j - 1], fine[0][2 * j]);
        coarse[Nx][j] = std::max(fine[2 * Nx][2 * j - 1], fine[2 * Nx][2 * j]);
    }
    return coarse;
}

template <typename T>
vector2D<T> interpolate(const vector2D<T>& coarse, int Nx, int Ny) {
    auto fine = make_vector_2D<T>(2 * Nx + 1, 2 * Ny + 2);
    for (size_t i = 0; i <= Nx; i++) {
        for (size_t j = 0; j <= Ny; j++) {
            auto a                 = coarse[i][j];
            auto b                 = coarse[i][j + 1];
            auto c                 = 0.25 * (b - a);
            fine[2 * i][2 * j]     = a + c;
            fine[2 * i][2 * j + 1] = a + 3 * c;
        }
    }
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j <= Ny; j++) {
            fine[2 * i + 1][2 * j]     = 0.5 * (fine[2 * i][2 * j] + fine[2 * i + 2][2 * j]);
            fine[2 * i + 1][2 * j + 1] = 0.5 * (fine[2 * i][2 * j + 1] + fine[2 * i + 2][2 * j + 1]);
        }
    }
    return fine;
}
} // namespace u
} // namespace mg

int test_u() {
    int Nx = 4;
    int Ny = 4;

    auto coarse = mg::make_vector_2D<double>(Nx + 1, Ny + 2);
    auto fine   = mg::make_vector_2D<double>(2 * Nx + 1, 2 * Ny + 2);

    for (int i = 0; i <= 2 * Nx; ++i) {
        for (int j = 0; j <= 2 * Ny + 1; ++j) {
            fine[i][j] = (i + 100 * j);
        }
    }

    for (int i = 0; i <= Nx; ++i) {
        for (int j = 0; j <= Ny + 1; ++j) {
            coarse[i][j] = (i + 100 * j) * 3;
        }
    }

    std::vector<std::vector<double>> coarse_restricted = mg::u::restrict(fine, Nx, Ny);
    mg::print_vector_2D(fine);
    mg::print_vector_2D(coarse);

    coarse = mg::u::restrict_bc(fine, Nx, Ny);
    mg::print_vector_2D(coarse);

    for (int i = 0; i <= Nx; ++i) {
        for (int j = 0; j <= Ny + 1; ++j) {
            coarse[i][j] = (i + 100 * j) * 3;
        }
    }

    fine = mg::u::interpolate(coarse, Nx, Ny);
    mg::print_vector_2D(fine);

    return 0;
}

int main() { test_u(); }