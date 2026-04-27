/// @date 2023-10-01
/// @file VelocityU3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei

#ifndef __VELOCITY_U_3D_H__
#define __VELOCITY_U_3D_H__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

namespace velocity_u_3D {

// Define constants for boundary types
const int DIRICHLET = 1;
const int NEUMANN   = 2;
const int DIM       = 3;

using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

MultiArrayDouble generate_u_prime(const MultiArrayInt& boundary_type, const MultiArrayDouble& boundary_values,
                                  const MultiArrayDouble& bu, int Nx, int Ny, int Nz, double width, double height,
                                  double depth, double mu, double rho, double dt) {
    double           dx   = width / Nx;
    double           dy   = height / Ny;
    double           dz   = depth / Nz;
    MultiArrayDouble data = algebra::create_multi_array<3, double>({Nx + 3, Ny + 2, Nz + 2});

    for (int i = 1; i < Nx + 2; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i][0][k + 1] == DIRICHLET) {
                data[i][0][k + 1] = 2 * boundary_values[i][0][k + 1] - data[i][1][k + 1];
            }
            if (boundary_type[i][0][k + 1] == NEUMANN) {
                data[i][0][k + 1] = data[i][1][k + 1] - dy * boundary_values[i][0][k + 1];
            }
            if (boundary_type[i][Ny + 1][k + 1] == DIRICHLET) {
                data[i][Ny + 1][k + 1] = 2 * boundary_values[i][Ny + 1][k + 1] - data[i][Ny][k + 1];
            }
            if (boundary_type[i][Ny + 1][k + 1] == NEUMANN) {
                data[i][Ny + 1][k + 1] = data[i][Ny][k + 1] + dy * boundary_values[i][Ny + 1][k + 1];
            }
        }
    }

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i][j + 1][0] == DIRICHLET) {
                data[i][j + 1][0] = 2 * boundary_values[i][j + 1][0] - data[i][j + 1][1];
            }
            if (boundary_type[i][j + 1][0] == NEUMANN) {
                data[i][j + 1][0] = data[i][j + 1][1] - dz * boundary_values[i][j + 1][0];
            }
            if (boundary_type[i][j + 1][Nz + 1] == DIRICHLET) {
                data[i][j + 1][Nz + 1] = 2 * boundary_values[i][j + 1][Nz + 1] - data[i][j + 1][Nz];
            }
            if (boundary_type[i][j + 1][Nz + 1] == NEUMANN) {
                data[i][j + 1][Nz + 1] = data[i][j + 1][Nz] + dz * boundary_values[i][j + 1][Nz + 1];
            }
        }
    }

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[1][j + 1][k + 1] == DIRICHLET) {
                data[1][j + 1][k + 1] = boundary_values[1][j + 1][k + 1];
            }
            if (boundary_type[1][j + 1][k + 1] == NEUMANN) {
                data[0][j + 1][k + 1] = data[2][j + 1][k + 1] - 2.0 * dx * boundary_values[1][j + 1][k + 1];
            }
            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                data[Nx + 1][j + 1][k + 1] = boundary_values[Nx + 1][j + 1][k + 1];
            }
            if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                data[Nx + 2][j + 1][k + 1] = 2 * dx * boundary_values[Nx + 1][j + 1][k + 1] + data[Nx][j + 1][k + 1];
            }
        }
    }
    return data;
}

void smooth(MultiArrayDouble& u, const MultiArrayDouble& bu, const MultiArrayInt& boundary_type, int Nx, int Ny, int Nz,
            double width, double height, double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[1][j + 1][k + 1] == DIRICHLET) { u[1][j + 1][k + 1] = 0; }
            if (boundary_type[1][j + 1][k + 1] == NEUMANN) { u[0][j + 1][k + 1] = u[2][j + 1][k + 1]; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) { u[Nx + 1][j + 1][k + 1] = 0; }
            if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) { u[Nx + 2][j + 1][k + 1] = u[Nx][j + 1][k + 1]; }
        }
    }
    for (int i = 1; i < Nx + 2; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i][0][k + 1] == DIRICHLET) { u[i][0][k + 1] = -u[i][1][k + 1]; }
            if (boundary_type[i][0][k + 1] == NEUMANN) { u[i][0][k + 1] = u[i][1][k + 1]; }
            if (boundary_type[i][Ny + 1][k + 1] == DIRICHLET) { u[i][Ny + 1][k + 1] = -u[i][Ny][k + 1]; }
            if (boundary_type[i][Ny + 1][k + 1] == NEUMANN) { u[i][Ny + 1][k + 1] = u[i][Ny][k + 1]; }
        }
    }

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i][j + 1][0] == DIRICHLET) { u[i][j + 1][0] = -u[i][j + 1][1]; }
            if (boundary_type[i][j + 1][0] == NEUMANN) { u[i][j + 1][0] = u[i][j + 1][1]; }
            if (boundary_type[i][j + 1][Nz + 1] == DIRICHLET) { u[i][j + 1][Nz + 1] = -u[i][j + 1][Nz]; }
            if (boundary_type[i][j + 1][Nz + 1] == NEUMANN) { u[i][j + 1][Nz + 1] = u[i][j + 1][Nz]; }
        }
    }

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i][j + 1][k + 1] == DIRICHLET) continue;
                if ((i + j + k) % 2 == 0) {
                    double b1          = (u[i - 1][j + 1][k + 1] + u[i + 1][j + 1][k + 1]) / (dx * dx);
                    double b2          = (u[i][j][k + 1] + u[i][j + 2][k + 1]) / (dy * dy);
                    double b3          = (u[i][j + 1][k] + u[i][j + 1][k + 2]) / (dz * dz);
                    double b           = mu * (b1 + b2 + b3);
                    double a           = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                    u[i][j + 1][k + 1] = (bu[i][j + 1][k + 1] + b) / a;
                }
            }
        }
    }

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i][j + 1][k + 1] == DIRICHLET) continue;
                if ((i + j + k) % 2 == 1) {
                    double b1          = (u[i - 1][j + 1][k + 1] + u[i + 1][j + 1][k + 1]) / (dx * dx);
                    double b2          = (u[i][j][k + 1] + u[i][j + 2][k + 1]) / (dy * dy);
                    double b3          = (u[i][j + 1][k] + u[i][j + 1][k + 2]) / (dz * dz);
                    double b           = mu * (b1 + b2 + b3);
                    double a           = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                    u[i][j + 1][k + 1] = (bu[i][j + 1][k + 1] + b) / a;
                }
            }
        }
    }
}

MultiArrayDouble residual_prime(MultiArrayDouble& u, const MultiArrayDouble& bu, const MultiArrayInt& boundary_type,
                                const MultiArrayDouble& boundary_values, int Nx, int Ny, int Nz, double width,
                                double height, double depth, double mu, double rho, double dt) {
    double           dx = width / Nx;
    double           dy = height / Ny;
    double           dz = depth / Nz;
    MultiArrayDouble r  = algebra::create_multi_array<3, double>({Nx + 3, Ny + 2, Nz + 2});

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i][j + 1][k + 1] == DIRICHLET) continue;
                double b1          = (u[i - 1][j + 1][k + 1] + u[i + 1][j + 1][k + 1]) / (dx * dx);
                double b2          = (u[i][j][k + 1] + u[i][j + 2][k + 1]) / (dy * dy);
                double b3          = (u[i][j + 1][k] + u[i][j + 1][k + 2]) / (dz * dz);
                double b           = mu * (b1 + b2 + b3);
                double a           = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                r[i][j + 1][k + 1] = bu[i][j + 1][k + 1] + b - a * u[i][j + 1][k + 1];
            }
        }
    }

    return r;
}

void residual(MultiArrayDouble& r, MultiArrayDouble& u, const MultiArrayDouble& bu, const MultiArrayInt& boundary_type,
              int Nx, int Ny, int Nz, double width, double height, double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[1][j + 1][k + 1] == DIRICHLET) {
                u[1][j + 1][k + 1] = 0.0;
            } else if (boundary_type[1][j + 1][k + 1] == NEUMANN) {
                u[0][j + 1][k + 1] = u[2][j + 1][k + 1];
            }

            if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                u[Nx + 1][j + 1][k + 1] = 0.0;
            } else if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                u[Nx + 2][j + 1][k + 1] = u[Nx][j + 1][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx + 2; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i][0][k + 1] == DIRICHLET) {
                u[i][0][k + 1] = -u[i][1][k + 1];
            } else if (boundary_type[i][0][k + 1] == NEUMANN) {
                u[i][0][k + 1] = u[i][1][k + 1];
            }

            if (boundary_type[i][Ny + 1][k + 1] == DIRICHLET) {
                u[i][Ny + 1][k + 1] = -u[i][Ny][k + 1];
            } else if (boundary_type[i][Ny + 1][k + 1] == NEUMANN) {
                u[i][Ny + 1][k + 1] = u[i][Ny][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i][j + 1][0] == DIRICHLET) {
                u[i][j + 1][0] = -u[i][j + 1][1];
            } else if (boundary_type[i][j + 1][0] == NEUMANN) {
                u[i][j + 1][0] = u[i][j + 1][1];
            }

            if (boundary_type[i][j + 1][Nz + 1] == DIRICHLET) {
                u[i][j + 1][Nz + 1] = -u[i][j + 1][Nz];
            } else if (boundary_type[i][j + 1][Nz + 1] == NEUMANN) {
                u[i][j + 1][Nz + 1] = u[i][j + 1][Nz];
            }
        }
    }

    for (int i = 1; i < Nx + 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i][j + 1][k + 1] == DIRICHLET) continue;
                double b1          = (u[i - 1][j + 1][k + 1] + u[i + 1][j + 1][k + 1]) / (dx * dx);
                double b2          = (u[i][j][k + 1] + u[i][j + 2][k + 1]) / (dy * dy);
                double b3          = (u[i][j + 1][k] + u[i][j + 1][k + 2]) / (dz * dz);
                double b           = mu * (b1 + b2 + b3);
                double a           = rho / dt + 2 * mu * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz));
                r[i][j + 1][k + 1] = bu[i][j + 1][k + 1] + b - a * u[i][j + 1][k + 1];
            }
        }
    }
}

void solve_u(MultiArrayDouble& rh, MultiArrayDouble& uh, const MultiArrayInt& ubt, const MultiArrayDouble& ubv,
             const MultiArrayDouble& bu, int Nx, int Ny, int Nz, double width, double height, double depth, double mu,
             double rho, double dt, int max_iters, double tolerance) {
    ScopeProfiler _{__func__};

    double dx      = width / Nx;
    double dy      = height / Ny;
    double dz      = depth / Nz;
    auto   u_prime = generate_u_prime(ubt, ubv, bu, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
    auto   bu_hat  = residual_prime(u_prime, bu, ubt, ubv, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
    for (int iter = 0; iter < max_iters; iter++) {
        smooth(uh, bu_hat, ubt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
        residual(rh, uh, bu_hat, ubt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
        double sum_r_squared = algebra::norm(rh) * std::sqrt(dx * dy * dz);
        printf("iter = %d, sum_r_squared  = %.20e\n", iter, sum_r_squared);
        if (sum_r_squared < tolerance) { break; }
    }
    algebra::axpy(1.0, u_prime, uh);
}

} // namespace velocity_u_3D

#endif