
#include "mglib.h"

namespace mg {
std::tuple<int, std::vector<LevelInfo>> compute_multigrid_levels_2D(int Nx, int Ny, double width, double height) {
    int                    num_levels = 0;
    std::vector<LevelInfo> level_info;
    double                 dx = width / Nx;
    double                 dy = height / Ny;

    while (true) {
        num_levels++;
        level_info.push_back({dx, dy, Nx, Ny});

        printf("num_levels : %d, level_dh  : %2.4e, %2.4e\n", num_levels, dx, dy);
        printf("num_levels : %d, level_dim : %d, %d\n", num_levels, Nx, Ny);

        if (Nx % 2 == 1 || Ny % 2 == 1) { break; }

        if (Nx * Ny < 17) { break; }

        Nx /= 2;
        Ny /= 2;
        dx *= 2;
        dy *= 2;
    }

    if (num_levels <= 1) { throw std::runtime_error("the problem is too small for the multigrid solver."); }

    return {num_levels, level_info};
}

vector2D<double> restrict(const vector2D<double>& fine, int Nx, int Ny) {
    auto coarse = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (int i = 1; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            coarse[i][j]
                = 0.25
                  * (fine[2 * i - 1][2 * j - 1] + fine[2 * i][2 * j - 1] + fine[2 * i - 1][2 * j] + fine[2 * i][2 * j]);
        }
    }
    return coarse;
}

vector2D<int> restrict_bc(const vector2D<int>& fine, int Nx, int Ny) {
    vector2D<int> coarse = make_vector_2D<int>(Nx + 2, Ny + 2);

    for (int i = 1; i <= Nx; ++i) {
        coarse[i][0]      = std::max(fine[2 * i - 1][0], fine[2 * i][0]);
        coarse[i][Ny + 1] = std::max(fine[2 * i - 1][2 * Ny + 1], fine[2 * i][2 * Ny + 1]);
    }

    for (int j = 1; j <= Ny; ++j) {
        coarse[0][j]      = std::max(fine[0][2 * j - 1], fine[0][2 * j]);
        coarse[Nx + 1][j] = std::max(fine[2 * Nx + 1][2 * j - 1], fine[2 * Nx + 1][2 * j]);
    }

    return coarse;
}

double L0(double x) { return 1.0 - x; }
double L1(double x) { return x; }
double (*L[2])(double) = {L0, L1};

double bilinear(double x, double y, const std::vector<double>& f) {
    double sum = 0.0;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            sum += f[j * 2 + i] * L[i](x) * L[j](y);
        }
    }

    return sum;
}

vector2D<double> interpolate(const vector2D<double>& coarse, int Nx, int Ny) {
    vector2D<double> fine(2 * Nx + 2, std::vector<double>(2 * Ny + 2, 0.0));

    for (int i = 0; i <= Nx; ++i) {
        for (int j = 0; j <= Ny; ++j) {
            std::vector<double> f = {coarse[i][j], coarse[i + 1][j], coarse[i][j + 1], coarse[i + 1][j + 1]};

            fine[2 * i][2 * j]         = bilinear(1.0 / 3.0, 1.0 / 3.0, f);
            fine[2 * i][2 * j + 1]     = bilinear(1.0 / 3.0, 2.0 / 3.0, f);
            fine[2 * i + 1][2 * j]     = bilinear(2.0 / 3.0, 1.0 / 3.0, f);
            fine[2 * i + 1][2 * j + 1] = bilinear(2.0 / 3.0, 2.0 / 3.0, f);
        }
    }

    return fine;
}

std::vector<std::vector<double>> compute_residual(std::vector<std::vector<double>>&       phi,
                                                  const std::vector<std::vector<int>>&    boundary_type,
                                                  const std::vector<std::vector<double>>& b, int Nx, int Ny,
                                                  double width, double height) {
    double dx = width / Nx;
    double dy = height / Ny;

    auto residual = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (int i = 1; i <= Nx; ++i) {
        if (boundary_type[i][0] == NEUMANN) { phi[i][0] = phi[i][1]; }
        if (boundary_type[i][Ny + 1] == NEUMANN) { phi[i][Ny + 1] = phi[i][Ny]; }
    }

    for (int j = 1; j <= Ny; ++j) {
        if (boundary_type[0][j] == NEUMANN) { phi[0][j] = phi[1][j]; }
        if (boundary_type[Nx + 1][j] == NEUMANN) { phi[Nx + 1][j] = phi[Nx][j]; }
    }

    for (int i = 1; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            double p1      = (phi[i - 1][j] - 2 * phi[i][j] + phi[i + 1][j]) / dx / dx;
            double p2      = (phi[i][j - 1] - 2 * phi[i][j] + phi[i][j + 1]) / dy / dy;
            residual[i][j] = b[i][j] - p1 - p2;
        }
    }

    return residual;
}

std::vector<std::vector<double>> smooth(std::vector<std::vector<double>>&       phi,
                                        const std::vector<std::vector<int>>&    boundary_type,
                                        const std::vector<std::vector<double>>& b, int Nx, int Ny, double width,
                                        double height) {
    double dx = width / Nx;
    double dy = height / Ny;

    for (int i = 1; i <= Nx; ++i) {
        phi[i][0]      = (boundary_type[i][0] == DIRICHLET) ? (-phi[i][1])
                         : (boundary_type[i][0] == NEUMANN) ? phi[i][1]
                                                            : phi[i][0];
        phi[i][Ny + 1] = (boundary_type[i][Ny + 1] == DIRICHLET) ? (-phi[i][Ny])
                         : (boundary_type[i][Ny + 1] == NEUMANN) ? phi[i][Ny]
                                                                 : phi[i][Ny + 1];
    }

    for (int j = 1; j <= Ny; ++j) {
        phi[0][j]      = (boundary_type[0][j] == DIRICHLET) ? (-phi[1][j])
                         : (boundary_type[0][j] == NEUMANN) ? phi[1][j]
                                                            : phi[0][j];
        phi[Nx + 1][j] = (boundary_type[Nx + 1][j] == DIRICHLET) ? (-phi[Nx][j])
                         : (boundary_type[Nx + 1][j] == NEUMANN) ? phi[Nx][j]
                                                                 : phi[Nx + 1][j];
    }

    for (int i = 1; i <= Nx; ++i) {
        for (int j = 1; j <= Ny; ++j) {
            phi[i][j] = ((phi[i + 1][j] + phi[i - 1][j]) * dy * dy + (phi[i][j + 1] + phi[i][j - 1]) * dx * dx
                         - b[i][j] * dx * dx * dy * dy)
                        / (2 * dx * dx + 2 * dy * dy);
        }
    }

    return phi;
}

} // namespace mg
