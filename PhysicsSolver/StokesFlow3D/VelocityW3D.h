/// @date 2023-10-03
/// @file VelocityW3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __VELOCITY_W_3D_H__
#define __VELOCITY_W_3D_H__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

namespace velocity_w_3D {

// Define constants for boundary types
const int DIRICHLET = 1;
const int NEUMANN   = 2;
const int DIM       = 3;

using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

MultiArrayDouble generate_u_prime(const MultiArrayInt& boundary_type, const MultiArrayDouble& boundary_values,
                                  const MultiArrayDouble& bw, int Nx, int Ny, int Nz, double width, double height,
                                  double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    MultiArrayDouble w = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 3});

    for (int j = 0; j < Ny; ++j) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[0][j + 1][k] == DIRICHLET) {
                w[0][j + 1][k] = -w[1][j + 1][k] + 2.0 * boundary_values[0][j + 1][k];
            }
            if (boundary_type[0][j + 1][k] == NEUMANN) {
                w[0][j + 1][k] = w[1][j + 1][k] - dx * boundary_values[0][j + 1][k];
            }
            if (boundary_type[Nx + 1][j + 1][k] == DIRICHLET) {
                w[Nx + 1][j + 1][k] = -w[Nx][j + 1][k] + 2.0 * boundary_values[Nx + 1][j + 1][k];
            }
            if (boundary_type[Nx + 1][j + 1][k] == NEUMANN) {
                w[Nx + 1][j + 1][k] = w[Nx][j + 1][k] + dx * boundary_values[Nx + 1][j + 1][k];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[i + 1][0][k] == DIRICHLET) {
                w[i + 1][0][k] = -w[i + 1][1][k] + 2.0 * boundary_values[i + 1][0][k];
            }
            if (boundary_type[i + 1][0][k] == NEUMANN) {
                w[i + 1][0][k] = w[i + 1][1][k] - dy * boundary_values[i + 1][0][k];
            }
            if (boundary_type[i + 1][Ny + 1][k] == DIRICHLET) {
                w[i + 1][Ny + 1][k] = -w[i + 1][Ny][k] + 2.0 * boundary_values[i + 1][Ny + 1][k];
            }
            if (boundary_type[i + 1][Ny + 1][k] == NEUMANN) {
                w[i + 1][Ny + 1][k] = w[i + 1][Ny][k] + dy * boundary_values[i + 1][Ny + 1][k];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][1] == DIRICHLET) { w[i + 1][j + 1][1] = boundary_values[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][1] == NEUMANN) {
                w[i + 1][j + 1][0] = w[i + 1][j + 1][2] - 2.0 * dz * boundary_values[i + 1][j + 1][1];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                w[i + 1][j + 1][Nz + 1] = boundary_values[i + 1][j + 1][Nz + 1];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                w[i + 1][j + 1][Nz + 2] = w[i + 1][j + 1][Nz] + 2.0 * dz * boundary_values[i + 1][j + 1][Nz + 1];
            }
        }
    }

    return w;
}

MultiArrayDouble smooth(MultiArrayDouble& w, const MultiArrayDouble& bw, const MultiArrayInt& boundary_type, int Nx,
                        int Ny, int Nz, double width, double height, double depth, double mu, double rho, double dt) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][1] == DIRICHLET) { w[i + 1][j + 1][1] = 0.0; }
            if (boundary_type[i + 1][j + 1][1] == NEUMANN) { w[i + 1][j + 1][0] = w[i + 1][j + 1][2]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) { w[i + 1][j + 1][Nz + 1] = 0.0; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) { w[i + 1][j + 1][Nz + 2] = w[i + 1][j + 1][Nz]; }
        }
    }

    for (int j = 0; j < Ny; ++j) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[0][j + 1][k] == DIRICHLET) { w[0][j + 1][k] = -w[1][j + 1][k]; }
            if (boundary_type[0][j + 1][k] == NEUMANN) { w[0][j + 1][k] = w[1][j + 1][k]; }
            if (boundary_type[Nx + 1][j + 1][k] == DIRICHLET) { w[Nx + 1][j + 1][k] = -w[Nx][j + 1][k]; }
            if (boundary_type[Nx + 1][j + 1][k] == NEUMANN) { w[Nx + 1][j + 1][k] = w[Nx][j + 1][k]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[i + 1][0][k] == DIRICHLET) { w[i + 1][0][k] = -w[i + 1][1][k]; }
            if (boundary_type[i + 1][0][k] == NEUMANN) { w[i + 1][0][k] = w[i + 1][1][k]; }
            if (boundary_type[i + 1][Ny + 1][k] == DIRICHLET) { w[i + 1][Ny + 1][k] = -w[i + 1][Ny][k]; }
            if (boundary_type[i + 1][Ny + 1][k] == NEUMANN) { w[i + 1][Ny + 1][k] = w[i + 1][Ny][k]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 1; k < Nz + 2; ++k) {
                if (boundary_type[i + 1][j + 1][k] == DIRICHLET) continue;
                double b1          = (w[i][j + 1][k] + w[i + 2][j + 1][k]) / (dx * dx);
                double b2          = (w[i + 1][j][k] + w[i + 1][j + 2][k]) / (dy * dy);
                double b3          = (w[i + 1][j + 1][k - 1] + w[i + 1][j + 1][k + 1]) / (dz * dz);
                double b           = mu * (b1 + b2 + b3);
                double a           = rho / dt + 2 * mu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                w[i + 1][j + 1][k] = (bw[i + 1][j + 1][k] + b) / a;
            }
        }
    }

    return w;
}

MultiArrayDouble residual_prime(MultiArrayDouble& w, const MultiArrayDouble& bw, const MultiArrayInt& boundary_type,
                                const MultiArrayDouble& boundary_values, int Nx, int Ny, int Nz, double width,
                                double height, double depth, double mu, double rho, double dt) {
    double           dx = width / Nx;
    double           dy = height / Ny;
    double           dz = depth / Nz;
    MultiArrayDouble r  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 3});

    for (int j = 0; j < Ny; ++j) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[0][j + 1][k] == DIRICHLET) {
                w[0][j + 1][k] = -w[1][j + 1][k] + 2.0 * boundary_values[0][j + 1][k];
            }
            if (boundary_type[0][j + 1][k] == NEUMANN) {
                w[0][j + 1][k] = w[1][j + 1][k] - dx * boundary_values[0][j + 1][k];
            }
            if (boundary_type[Nx + 1][j + 1][k] == DIRICHLET) {
                w[Nx + 1][j + 1][k] = -w[Nx][j + 1][k] + 2.0 * boundary_values[Nx + 1][j + 1][k];
            }
            if (boundary_type[Nx + 1][j + 1][k] == NEUMANN) {
                w[Nx + 1][j + 1][k] = w[Nx][j + 1][k] + dx * boundary_values[Nx + 1][j + 1][k];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[i + 1][0][k] == DIRICHLET) {
                w[i + 1][0][k] = -w[i + 1][1][k] + 2.0 * boundary_values[i + 1][0][k];
            }
            if (boundary_type[i + 1][0][k] == NEUMANN) {
                w[i + 1][0][k] = w[i + 1][1][k] - dy * boundary_values[i + 1][0][k];
            }
            if (boundary_type[i + 1][Ny + 1][k] == DIRICHLET) {
                w[i + 1][Ny + 1][k] = -w[i + 1][Ny][k] + 2.0 * boundary_values[i + 1][Ny + 1][k];
            }
            if (boundary_type[i + 1][Ny + 1][k] == NEUMANN) {
                w[i + 1][Ny + 1][k] = w[i + 1][Ny][k] + dy * boundary_values[i + 1][Ny + 1][k];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][1] == DIRICHLET) { w[i + 1][j + 1][1] = boundary_values[i + 1][j + 1][1]; }
            if (boundary_type[i + 1][j + 1][1] == NEUMANN) {
                w[i + 1][j + 1][0] = w[i + 1][j + 1][2] - 2.0 * dz * boundary_values[i + 1][j + 1][1];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                w[i + 1][j + 1][Nz + 1] = boundary_values[i + 1][j + 1][Nz + 1];
            }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                w[i + 1][j + 1][Nz + 2] = w[i + 1][j + 1][Nz] + 2.0 * dz * boundary_values[i + 1][j + 1][Nz + 1];
            }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 1; k < Nz + 2; ++k) {
                if (boundary_type[i + 1][j + 1][k] == DIRICHLET) continue;
                double b1 = (w[i][j + 1][k] - 2.0 * w[i + 1][j + 1][k] + w[i + 2][j + 1][k]) / (dx * dx);
                double b2 = (w[i + 1][j][k] - 2.0 * w[i + 1][j + 1][k] + w[i + 1][j + 2][k]) / (dy * dy);
                double b3 = (w[i + 1][j + 1][k - 1] - 2.0 * w[i + 1][j + 1][k] + w[i + 1][j + 1][k + 1]) / (dz * dz);
                r[i + 1][j + 1][k] = bw[i + 1][j + 1][k] - rho / dt * w[i + 1][j + 1][k] + mu * (b1 + b2 + b3);
            }
        }
    }

    return r;
}

MultiArrayDouble residual(MultiArrayDouble& w, const MultiArrayDouble& bw, const MultiArrayInt& boundary_type, int Nx,
                          int Ny, int Nz, double width, double height, double depth, double mu, double rho, double dt) {
    double           dx = width / Nx;
    double           dy = height / Ny;
    double           dz = depth / Nz;
    MultiArrayDouble r  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 3});
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            if (boundary_type[i + 1][j + 1][1] == DIRICHLET) { w[i + 1][j + 1][1] = 0.0; }
            if (boundary_type[i + 1][j + 1][1] == NEUMANN) { w[i + 1][j + 1][0] = w[i + 1][j + 1][2]; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) { w[i + 1][j + 1][Nz + 1] = 0.0; }
            if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) { w[i + 1][j + 1][Nz + 2] = w[i + 1][j + 1][Nz]; }
        }
    }

    for (int j = 0; j < Ny; ++j) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[0][j + 1][k] == DIRICHLET) { w[0][j + 1][k] = -w[1][j + 1][k]; }
            if (boundary_type[0][j + 1][k] == NEUMANN) { w[0][j + 1][k] = w[1][j + 1][k]; }
            if (boundary_type[Nx + 1][j + 1][k] == DIRICHLET) { w[Nx + 1][j + 1][k] = -w[Nx][j + 1][k]; }
            if (boundary_type[Nx + 1][j + 1][k] == NEUMANN) { w[Nx + 1][j + 1][k] = w[Nx][j + 1][k]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int k = 1; k < Nz + 2; ++k) {
            if (boundary_type[i + 1][0][k] == DIRICHLET) { w[i + 1][0][k] = -w[i + 1][1][k]; }
            if (boundary_type[i + 1][0][k] == NEUMANN) { w[i + 1][0][k] = w[i + 1][1][k]; }
            if (boundary_type[i + 1][Ny + 1][k] == DIRICHLET) { w[i + 1][Ny + 1][k] = -w[i + 1][Ny][k]; }
            if (boundary_type[i + 1][Ny + 1][k] == NEUMANN) { w[i + 1][Ny + 1][k] = w[i + 1][Ny][k]; }
        }
    }

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 1; k < Nz + 2; ++k) {
                if (boundary_type[i + 1][j + 1][k] == DIRICHLET) continue;
                double b1 = (w[i][j + 1][k] - 2.0 * w[i + 1][j + 1][k] + w[i + 2][j + 1][k]) / (dx * dx);
                double b2 = (w[i + 1][j][k] - 2.0 * w[i + 1][j + 1][k] + w[i + 1][j + 2][k]) / (dy * dy);
                double b3 = (w[i + 1][j + 1][k - 1] - 2.0 * w[i + 1][j + 1][k] + w[i + 1][j + 1][k + 1]) / (dz * dz);
                r[i + 1][j + 1][k] = bw[i + 1][j + 1][k] - rho / dt * w[i + 1][j + 1][k] + mu * (b1 + b2 + b3);
            }
        }
    }

    return r;
}
void solve_w(MultiArrayDouble& rh, MultiArrayDouble& wh, const MultiArrayInt& wbt, const MultiArrayDouble& wbv,
             const MultiArrayDouble& bw, int Nx, int Ny, int Nz, double width, double height, double depth, double mu,
             double rho, double dt, int max_iters, double tolerance) {
    ScopeProfiler _{__func__};
    double        dx = width / Nx;
    double        dy = height / Ny;
    double        dz = depth / Nz;

    auto w_prime = velocity_w_3D::generate_u_prime(wbt, wbv, bw, Nx, Ny, Nz, width, height, depth, mu, rho, dt);
    auto bw_hat  = velocity_w_3D::residual_prime(w_prime, bw, wbt, wbv, Nx, Ny, Nz, width, height, depth, mu, rho, dt);

    for (int iter = 0; iter < max_iters; iter++) {
        wh = velocity_w_3D::smooth(wh, bw_hat, wbt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);

        auto rh = velocity_w_3D::residual(wh, bw_hat, wbt, Nx, Ny, Nz, width, height, depth, mu, rho, dt);

        double sum_r_squared = algebra::norm(rh) * std::sqrt(dx * dy * dz);
        // printf("iter = %d, sum_r_squared = %.20e\n", iter, sum_r_squared);
        if (sum_r_squared < tolerance) { break; }
    }

    algebra::axpy(1.0, w_prime, wh);
}

} // namespace velocity_w_3D

#endif