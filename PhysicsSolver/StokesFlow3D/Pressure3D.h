/// @date 2023-10-01
/// @file Pressure3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __PRESSURE_3D_H__
#define __PRESSURE_3D_H__

#include <PhysicsSolver/StokesFlow3D/some_headers.h>

namespace pressure_3D {
const int DIM = 3;

using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

MultiArrayDouble generate_r_prime(const MultiArrayDouble& p, const MultiArrayDouble& bp, int Nx, int Ny, int Nz,
                                  double width, double height, double depth) {
    double           dx = width / Nx;
    double           dy = height / Ny;
    double           dz = depth / Nz;
    MultiArrayDouble r(Nx + 2, std::vector<std::vector<double>>(Ny + 2, std::vector<double>(Nz + 2, 0.0)));

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double ra = (p[i + 2][j + 1][k + 1] - 2 * p[i + 1][j + 1][k + 1] + p[i][j + 1][k + 1]) / (dx * dx);
                double rb = (p[i + 1][j + 2][k + 1] - 2 * p[i + 1][j + 1][k + 1] + p[i + 1][j][k + 1]) / (dy * dy);
                double rc = (p[i + 1][j + 1][k + 2] - 2 * p[i + 1][j + 1][k + 1] + p[i + 1][j + 1][k]) / (dz * dz);
                r[i + 1][j + 1][k + 1] = bp[i + 1][j + 1][k + 1] - ra - rb - rc;
            }
        }
    }

    return r;
}

void smooth(MultiArrayDouble& p, const MultiArrayDouble& bp, const MultiArrayInt& boundary_type, int Nx, int Ny, int Nz,
            double width, double height, double depth) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j + 1][k + 1] == DIRICHLET) { p[0][j + 1][k + 1] = -p[1][j + 1][k + 1]; }
            if (boundary_type[0][j + 1][k + 1] == NEUMANN) { p[0][j + 1][k + 1] = p[1][j + 1][k + 1]; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) { p[Nx + 1][j + 1][k + 1] = -p[Nx][j + 1][k + 1]; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) { p[Nx + 1][j + 1][k + 1] = p[Nx][j + 1][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][0][k + 1] == DIRICHLET) { p[i + 1][0][k + 1] = -p[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][0][k + 1] == NEUMANN) { p[i + 1][0][k + 1] = p[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) { p[i + 1][Ny + 1][k + 1] = -p[i + 1][Ny][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) { p[i + 1][Ny + 1][k + 1] = p[i + 1][Ny][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][0] == DIRICHLET) { p[i + 1][j + 1][0] = -p[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][0] == NEUMANN) { p[i + 1][j + 1][0] = p[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) { p[i + 1][j + 1][Nz + 1] = -p[i + 1][j + 1][Nz]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) { p[i + 1][j + 1][Nz + 1] = p[i + 1][j + 1][Nz]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if ((i + j + k) % 2 == 0) {
                    double b1              = (p[i + 2][j + 1][k + 1] + p[i][j + 1][k + 1]) / (dx * dx);
                    double b2              = (p[i + 1][j + 2][k + 1] + p[i + 1][j][k + 1]) / (dy * dy);
                    double b3              = (p[i + 1][j + 1][k + 2] + p[i + 1][j + 1][k]) / (dz * dz);
                    double b               = b1 + b2 + b3;
                    double a               = -2 / (dx * dx) - 2 / (dy * dy) - 2 / (dz * dz);
                    p[i + 1][j + 1][k + 1] = (bp[i + 1][j + 1][k + 1] - b) / a;
                }
            }
        }
    }
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if ((i + j + k) % 2 == 1) {
                    double b1              = (p[i + 2][j + 1][k + 1] + p[i][j + 1][k + 1]) / (dx * dx);
                    double b2              = (p[i + 1][j + 2][k + 1] + p[i + 1][j][k + 1]) / (dy * dy);
                    double b3              = (p[i + 1][j + 1][k + 2] + p[i + 1][j + 1][k]) / (dz * dz);
                    double b               = b1 + b2 + b3;
                    double a               = -2 / (dx * dx) - 2 / (dy * dy) - 2 / (dz * dz);
                    p[i + 1][j + 1][k + 1] = (bp[i + 1][j + 1][k + 1] - b) / a;
                }
            }
        }
    }
    printf("p:%f\n", p[1][1][1]);
}

// Function to calculate residual
void residual(MultiArrayDouble& r, MultiArrayDouble& p, const MultiArrayDouble& bp, const MultiArrayInt& boundary_type,
              int Nx, int Ny, int Nz, double width, double height, double depth) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j + 1][k + 1] == DIRICHLET) { p[0][j + 1][k + 1] = -p[1][j + 1][k + 1]; }
            if (boundary_type[0][j + 1][k + 1] == NEUMANN) { p[0][j + 1][k + 1] = p[1][j + 1][k + 1]; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) { p[Nx + 1][j + 1][k + 1] = -p[Nx][j + 1][k + 1]; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) { p[Nx + 1][j + 1][k + 1] = p[Nx][j + 1][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][0][k + 1] == DIRICHLET) { p[i + 1][0][k + 1] = -p[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][0][k + 1] == NEUMANN) { p[i + 1][0][k + 1] = p[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) { p[i + 1][Ny + 1][k + 1] = -p[i + 1][Ny][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) { p[i + 1][Ny + 1][k + 1] = p[i + 1][Ny][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][0] == DIRICHLET) { p[i + 1][j + 1][0] = -p[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][0] == NEUMANN) { p[i + 1][j + 1][0] = p[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) { p[i + 1][j + 1][Nz + 1] = -p[i + 1][j + 1][Nz]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) { p[i + 1][j + 1][Nz + 1] = p[i + 1][j + 1][Nz]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double ra = (p[i + 2][j + 1][k + 1] - 2 * p[i + 1][j + 1][k + 1] + p[i][j + 1][k + 1]) / (dx * dx);
                double rb = (p[i + 1][j + 2][k + 1] - 2 * p[i + 1][j + 1][k + 1] + p[i + 1][j][k + 1]) / (dy * dy);
                double rc = (p[i + 1][j + 1][k + 2] - 2 * p[i + 1][j + 1][k + 1] + p[i + 1][j + 1][k]) / (dz * dz);
                r[i + 1][j + 1][k + 1] = bp[i + 1][j + 1][k + 1] - ra - rb - rc;
            }
        }
    }
}

// Function to generate p_prime
MultiArrayDouble generate_p_prime(const MultiArrayInt& boundary_type, const MultiArrayDouble& boundary_values, int Nx,
                                  int Ny, int Nz, double width, double height, double depth) {
    double           dx   = width / Nx;
    double           dy   = height / Ny;
    double           dz   = depth / Nz;
    MultiArrayDouble data = algebra::create_multi_array<DIM, double>({Nx + 2, Ny + 2, Nz + 2});

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                data[0][j + 1][k + 1] = 2 * boundary_values[0][j + 1][k + 1] - data[1][j + 1][k + 1];
            }
            if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                data[0][j + 1][k + 1] = data[1][j + 1][k + 1] - dx * boundary_values[0][j + 1][k + 1];
            }
            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                data[Nx + 1][j + 1][k + 1] = 2 * boundary_values[Nx + 1][j + 1][k + 1] - data[Nx][j + 1][k + 1];
            }
            if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                data[Nx + 1][j + 1][k + 1] = data[Nx][j + 1][k + 1] + dx * boundary_values[Nx + 1][j + 1][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][0][k + 1] == DIRICHLET) {
                data[i + 1][0][k + 1] = 2 * boundary_values[i + 1][0][k + 1] - data[i + 1][1][k + 1];
            }
            if (boundary_type[i + 1][0][k + 1] == NEUMANN) {
                data[i + 1][0][k + 1] = data[i + 1][1][k + 1] - dy * boundary_values[i + 1][0][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                data[i + 1][Ny + 1][k + 1] = 2 * boundary_values[i + 1][Ny + 1][k + 1] - data[i + 1][Ny][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                data[i + 1][Ny + 1][k + 1] = data[i + 1][Ny][k + 1] + dy * boundary_values[i + 1][Ny + 1][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][0] == DIRICHLET) {
                data[i + 1][j + 1][0] = 2 * boundary_values[i + 1][j + 1][0] - data[i + 1][j + 1][1];
            }
            if (boundary_type[i + 1][j + 1][0] == NEUMANN) {
                data[i + 1][j + 1][0] = data[i + 1][j + 1][1] - dz * boundary_values[i + 1][j + 1][0];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                data[i + 1][j + 1][Nz + 1] = 2 * boundary_values[i + 1][j + 1][Nz + 1] - data[i + 1][j + 1][Nz];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                data[i + 1][j + 1][Nz + 1] = data[i + 1][j + 1][Nz] + dz * boundary_values[i + 1][j + 1][Nz + 1];
            }
        }
    }

    return data;
}

void solve_p(MultiArrayDouble& ph, MultiArrayDouble& r, const MultiArrayInt& pbt, const MultiArrayDouble& pbv,
             const MultiArrayDouble& bp, int Nx, int Ny, int Nz, double width, double height, double depth,
             int max_iters, double tolerance) {
    ScopeProfiler _{__func__};

    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    auto p_prime = generate_p_prime(pbt, pbv, Nx, Ny, Nz, width, height, depth);
    auto r_prime = generate_r_prime(p_prime, bp, Nx, Ny, Nz, width, height, depth);

    for (int iter = 0; iter < max_iters; ++iter) {
        smooth(ph, r_prime, pbt, Nx, Ny, Nz, width, height, depth);
        residual(r, ph, r_prime, pbt, Nx, Ny, Nz, width, height, depth);

        // Check convergence condition
        double sum_r_squared = algebra::norm(r) * std::sqrt(dx * dy * dz);
        // printf("iter = %d, sum_r_squared = %.20e\n", iter, sum_r_squared);
        if (sum_r_squared < tolerance) { break; }
    }
}
} // namespace pressure_3D

#endif