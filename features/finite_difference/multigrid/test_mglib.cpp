#include "mglib.h"

using namespace mg;
int main_2() {
    auto data = read_vector_2D<double>("b_new_c++.txt");
    print_vector_2D(data, 14);
    auto data_1 = read_vector_2D<int>("boundary_type.txt");
    print_vector_2D(data_1, 4);

    write_vector_2D(data, "b_new_c++_.txt");
    write_vector_2D(data_1, "bc_c++_.txt");

    return 0;
}

int main_323() {
    int Nx = 4;
    int Ny = 4;

    // Create a sample fine grid and coarse grid with some values
    auto fine   = make_vector_2D<double>(2 * Nx + 2, 2 * Ny + 2);
    auto coarse = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (int i = 0; i < 2 * Nx + 2; ++i) {
        for (int j = 0; j < 2 * Ny + 2; ++j) {
            fine[i][j] = 3 * (j * (100) + i);
        }
    }
    for (int i = 0; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny + 2; ++j) {
            coarse[i][j] = 3 * (j * (100) + i);
        }
    }
    print_vector_2D(fine);
    print_vector_2D(coarse);

    coarse = restrict(fine, Nx, Ny);
    print_vector_2D(coarse);

    for (int i = 0; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny + 2; ++j) {
            coarse[i][j] = 3 * (j * (100) + i);
        }
    }

    fine = interpolate(coarse, Nx, Ny);
    print_vector_2D(fine);

    const int NEUMANN   = 2;
    const int DIRICHLET = 1;

    std::vector<int> all_boundary_type = {NEUMANN, DIRICHLET, NEUMANN, DIRICHLET};
    auto             boundary_type     = make_vector_2D<int>(2 * Nx + 2, 2 * Ny + 2);

    for (int i = 1; i <= 2 * Nx; ++i) {
        boundary_type[i][0]          = all_boundary_type[0];
        boundary_type[i][2 * Ny + 1] = all_boundary_type[1];
    }

    for (int j = 1; j <= 2 * Ny; ++j) {
        boundary_type[0][j]          = all_boundary_type[2];
        boundary_type[2 * Nx + 1][j] = all_boundary_type[3];
    }

    print_vector_2D(boundary_type);

    auto coarse_bc = restrict_bc(boundary_type, Nx, Ny);

    print_vector_2D(coarse_bc);

    return 0;
}

int main() {
    int    Nx     = 64;
    int    Ny     = 64;
    double width  = 1.0;
    double height = 1.0;

    try {
        auto result     = compute_multigrid_levels_2D(Nx, Ny, width, height);
        int  num_levels = std::get<0>(result);
        std::cout << "Number of levels: " << num_levels << std::endl;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }

    return 0;
}
