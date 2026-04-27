//// @date 2023-10-02
/// @file VelocityV3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __VELOCITY_V_3D_H__
#define __VELOCITY_V_3D_H__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

namespace velocity_v_3D {

// Define constants for boundary types
const int DIRICHLET = 1;
const int NEUMANN   = 2;
const int DIM       = 3;

using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

MultiArrayDouble generate_v_prime(const MultiArrayInt& boundary_type, const MultiArrayDouble& boundary_values,
                                  const MultiArrayDouble& bv, int Nx, int Ny, int Nz, double width, double height,
                                  double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    MultiArrayDouble v = algebra::create_multi_array<3, double>({Nx + 2, Ny + 3, Nz + 2});

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            if (boundary_type[i + 1][j][0] == DIRICHLET) {
                v[i + 1][j][0] = 2 * boundary_values[i + 1][j][0] - v[i + 1][j][1];
            }
            if (boundary_type[i + 1][j][0] == NEUMANN) {
                v[i + 1][j][0] = v[i + 1][j][1] - dz * boundary_values[i + 1][j][0];
            }
            if (boundary_type[i + 1][j][Nz + 1] == DIRICHLET) {
                v[i + 1][j][Nz + 1] = 2 * boundary_values[i + 1][j][Nz + 1] - v[i + 1][j][Nz];
            }
            if (boundary_type[i + 1][j][Nz + 1] == NEUMANN) {
                v[i + 1][j][Nz + 1] = v[i + 1][j][Nz] + dz * boundary_values[i + 1][j][Nz + 1];
            }
        }
    }

    for (int j = 1; j < Ny + 2; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j][k + 1] == DIRICHLET) {
                v[0][j][k + 1] = 2 * boundary_values[0][j][k + 1] - v[1][j][k + 1];
            }
            if (boundary_type[0][j][k + 1] == NEUMANN) {
                v[0][j][k + 1] = v[1][j][k + 1] - dx * boundary_values[0][j][k + 1];
            }
            if (boundary_type[Nx + 1][j][k + 1] == DIRICHLET) {
                v[Nx + 1][j][k + 1] = 2 * boundary_values[Nx + 1][j][k + 1] - v[Nx][j][k + 1];
            }
            if (boundary_type[Nx + 1][j][k + 1] == NEUMANN) {
                v[Nx + 1][j][k + 1] = v[Nx][j][k + 1] + dx * boundary_values[Nx + 1][j][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][1][k + 1] == DIRICHLET) { v[i + 1][1][k + 1] = boundary_values[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][1][k + 1] == NEUMANN) {
                v[i + 1][0][k + 1] = v[i + 1][2][k + 1] - 2.0 * dy * boundary_values[i + 1][1][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                v[i + 1][Ny + 1][k + 1] = boundary_values[i + 1][Ny + 1][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                v[i + 1][Ny + 2][k + 1] = v[i + 1][Ny][k + 1] + 2.0 * dy * boundary_values[i + 1][Ny + 1][k + 1];
            }
        }
    }

    return v;
}

MultiArrayDouble smooth(MultiArrayDouble& v, const MultiArrayDouble& bv, const MultiArrayInt& boundary_type, int Nx,
                        int Ny, int Nz, double width, double height, double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][1][k + 1] == DIRICHLET) { v[i + 1][1][k + 1] = 0.0; }
            if (boundary_type[i + 1][1][k + 1] == NEUMANN) { v[i + 1][0][k + 1] = v[i + 1][2][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) { v[i + 1][Ny + 1][k + 1] = 0.0; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) { v[i + 1][Ny + 2][k + 1] = v[i + 1][Ny][k + 1]; }
        }
    }

    for (int j = 1; j < Ny + 2; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j][k + 1] == DIRICHLET) { v[0][j][k + 1] = -v[1][j][k + 1]; }
            if (boundary_type[0][j][k + 1] == NEUMANN) { v[0][j][k + 1] = v[1][j][k + 1]; }
            if (boundary_type[Nx + 1][j][k + 1] == DIRICHLET) { v[Nx + 1][j][k + 1] = -v[Nx][j][k + 1]; }
            if (boundary_type[Nx + 1][j][k + 1] == NEUMANN) { v[Nx + 1][j][k + 1] = v[Nx][j][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            if (boundary_type[i + 1][j][0] == DIRICHLET) { v[i + 1][j][0] = -v[i + 1][j][1]; }
            if (boundary_type[i + 1][j][0] == NEUMANN) { v[i + 1][j][0] = v[i + 1][j][1]; }
            if (boundary_type[i + 1][j][Nz + 1] == DIRICHLET) { v[i + 1][j][Nz + 1] = -v[i + 1][j][Nz]; }
            if (boundary_type[i + 1][j][Nz + 1] == NEUMANN) { v[i + 1][j][Nz + 1] = v[i + 1][j][Nz]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][j][k + 1] == DIRICHLET) continue;
                if ((i + j + k) % 2 == 0) {
                    double b1          = (v[i][j][k + 1] + v[i + 2][j][k + 1]) / (dx * dx);
                    double b2          = (v[i + 1][j - 1][k + 1] + v[i + 1][j + 1][k + 1]) / (dy * dy);
                    double b3          = (v[i + 1][j][k] + v[i + 1][j][k + 2]) / (dz * dz);
                    double b           = mu * (b1 + b2 + b3);
                    double a           = rho / dt + 2 * mu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                    v[i + 1][j][k + 1] = (bv[i + 1][j][k + 1] + b) / a;
                }
            }
        }
    }
    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][j][k + 1] == DIRICHLET) continue;
                if ((i + j + k) % 2 == 1) {
                    double b1          = (v[i][j][k + 1] + v[i + 2][j][k + 1]) / (dx * dx);
                    double b2          = (v[i + 1][j - 1][k + 1] + v[i + 1][j + 1][k + 1]) / (dy * dy);
                    double b3          = (v[i + 1][j][k] + v[i + 1][j][k + 2]) / (dz * dz);
                    double b           = mu * (b1 + b2 + b3);
                    double a           = rho / dt + 2 * mu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                    v[i + 1][j][k + 1] = (bv[i + 1][j][k + 1] + b) / a;
                }
            }
        }
    }
    return v;
}

MultiArrayDouble residual_prime(MultiArrayDouble& v, const MultiArrayDouble& bv, const MultiArrayInt& boundary_type,
                                const MultiArrayDouble& boundary_values, int Nx, int Ny, int Nz, double width,
                                double height, double depth, double mu, double rho, double dt) {
    double           dx = width / Nx;
    double           dy = height / Ny;
    double           dz = depth / Nz;
    MultiArrayDouble r  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 3, Nz + 2});

    for (int j = 1; j < Ny + 2; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j][k + 1] == DIRICHLET) {
                v[0][j][k + 1] = 2 * boundary_values[0][j][k + 1] - v[1][j][k + 1];
            }
            if (boundary_type[0][j][k + 1] == NEUMANN) {
                v[0][j][k + 1] = v[1][j][k + 1] - dx * boundary_values[0][j][k + 1];
            }
            if (boundary_type[Nx + 1][j][k + 1] == DIRICHLET) {
                v[Nx + 1][j][k + 1] = 2 * boundary_values[Nx + 1][j][k + 1] - v[Nx][j][k + 1];
            }
            if (boundary_type[Nx + 1][j][k + 1] == NEUMANN) {
                v[Nx + 1][j][k + 1] = v[Nx][j][k + 1] + dx * boundary_values[Nx + 1][j][k + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            if (boundary_type[i + 1][j][0] == DIRICHLET) {
                v[i + 1][j][0] = 2 * boundary_values[i + 1][j][0] - v[i + 1][j][1];
            }
            if (boundary_type[i + 1][j][0] == NEUMANN) {
                v[i + 1][j][0] = v[i + 1][j][1] - dz * boundary_values[i + 1][j][0];
            }
            if (boundary_type[i + 1][j][Nz + 1] == DIRICHLET) {
                v[i + 1][j][Nz + 1] = 2 * boundary_values[i + 1][j][Nz + 1] - v[i + 1][j][Nz];
            }
            if (boundary_type[i + 1][j][Nz + 1] == NEUMANN) {
                v[i + 1][j][Nz + 1] = v[i + 1][j][Nz] + dz * boundary_values[i + 1][j][Nz + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][1][k + 1] == DIRICHLET) { v[i + 1][1][k + 1] = boundary_values[i + 1][1][k + 1]; }
            if (boundary_type[i + 1][1][k + 1] == NEUMANN) {
                v[i + 1][0][k + 1] = v[i + 1][2][k + 1] - 2.0 * dy * boundary_values[i + 1][1][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                v[i + 1][Ny + 1][k + 1] = boundary_values[i + 1][Ny + 1][k + 1];
            }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                v[i + 1][Ny + 2][k + 1] = v[i + 1][Ny][k + 1] + 2.0 * dy * boundary_values[i + 1][Ny + 1][k + 1];
            }
        }
    }
    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][j][k + 1] == DIRICHLET) continue;
                double b1          = (v[i][j][k + 1] + v[i + 2][j][k + 1]) / (dx * dx);
                double b2          = (v[i + 1][j - 1][k + 1] + v[i + 1][j + 1][k + 1]) / (dy * dy);
                double b3          = (v[i + 1][j][k] + v[i + 1][j][k + 2]) / (dz * dz);
                double b           = mu * (b1 + b2 + b3);
                double a           = rho / dt + 2 * mu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                r[i + 1][j][k + 1] = bv[i + 1][j][k + 1] + b - a * v[i + 1][j][k + 1];
            }
        }
    }

    return r;
}

void residual(MultiArrayDouble& r, MultiArrayDouble& v, const MultiArrayDouble& bv, const MultiArrayInt& boundary_type,
              int Nx, int Ny, int Nz, double width, double height, double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;
    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[i + 1][1][k + 1] == DIRICHLET) { v[i + 1][1][k + 1] = 0.0; }
            if (boundary_type[i + 1][1][k + 1] == NEUMANN) { v[i + 1][0][k + 1] = v[i + 1][2][k + 1]; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) { v[i + 1][Ny + 1][k + 1] = 0.0; }
            if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) { v[i + 1][Ny + 2][k + 1] = v[i + 1][Ny][k + 1]; }
        }
    }
    for (int j = 1; j < Ny + 2; ++j) {
        for (int k = 0; k < Nz; ++k) {
            if (boundary_type[0][j][k + 1] == DIRICHLET) { v[0][j][k + 1] = -v[1][j][k + 1]; }
            if (boundary_type[0][j][k + 1] == NEUMANN) { v[0][j][k + 1] = v[1][j][k + 1]; }
            if (boundary_type[Nx + 1][j][k + 1] == DIRICHLET) { v[Nx + 1][j][k + 1] = -v[Nx][j][k + 1]; }
            if (boundary_type[Nx + 1][j][k + 1] == NEUMANN) { v[Nx + 1][j][k + 1] = v[Nx][j][k + 1]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            if (boundary_type[i + 1][j][0] == DIRICHLET) { v[i + 1][j][0] = -v[i + 1][j][1]; }
            if (boundary_type[i + 1][j][0] == NEUMANN) { v[i + 1][j][0] = v[i + 1][j][1]; }
            if (boundary_type[i + 1][j][Nz + 1] == DIRICHLET) { v[i + 1][j][Nz + 1] = -v[i + 1][j][Nz]; }
            if (boundary_type[i + 1][j][Nz + 1] == NEUMANN) { v[i + 1][j][Nz + 1] = v[i + 1][j][Nz]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 1; j < Ny + 2; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][j][k + 1] == DIRICHLET) continue;
                double b1          = (v[i][j][k + 1] + v[i + 2][j][k + 1]) / (dx * dx);
                double b2          = (v[i + 1][j - 1][k + 1] + v[i + 1][j + 1][k + 1]) / (dy * dy);
                double b3          = (v[i + 1][j][k] + v[i + 1][j][k + 2]) / (dz * dz);
                double b           = mu * (b1 + b2 + b3);
                double a           = rho / dt + 2 * mu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                r[i + 1][j][k + 1] = bv[i + 1][j][k + 1] + b - a * v[i + 1][j][k + 1];
            }
        }
    }

    // return r;
}
void solve_v(MultiArrayDouble& rh, MultiArrayDouble& vh, const MultiArrayInt& vbt, const MultiArrayDouble& vbv,
             const MultiArrayDouble& bv, int Nx, int Ny, int Nz, double width, double height, double depth, double mu,
             double rho, double dt, int max_iters, double tolerance) {
    ScopeProfiler _{__func__};

    double dx      = width / Nx;
    double dy      = height / Ny;
    double dz      = depth / Nz;
    auto   v_prime = generate_v_prime(vbt, vbv, bv, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
    auto   bv_hat  = residual_prime(v_prime, bv, vbt, vbv, Nx, Ny, Nz, width, height, depth, mu, rho, dt);

    for (int iter = 0; iter < max_iters; iter++) {
        smooth(vh, bv_hat, vbt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
        residual(rh, vh, bv_hat, vbt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);

        double sum_r_squared = algebra::norm(rh) * std::sqrt(dx * dy * dz);
        printf("iter = %d, sum_r_squared = %.20e\n", iter, sum_r_squared);
        if (sum_r_squared < tolerance) { break; }
    }
    algebra::axpy(1.0, v_prime, vh);
}

} // namespace velocity_v_3D

#endif