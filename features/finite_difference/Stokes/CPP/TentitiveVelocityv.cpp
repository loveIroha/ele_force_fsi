
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// using namespace std;

using std::exp;
using std::pow;

const int Nx = 200;
const int Ny = 200;
const int Nt = 200;

const int INTERIOR  = 0;
const int DIRICHLET = 1;
const int NEUMANN   = 2;

// const double EPSILON = std::sqrt(std::numeric_limits<double>::epsilon());
const double EPSILON = std::numeric_limits<double>::epsilon();

template <typename T>
auto make_vector_2D(int N, int M) {
    std::vector<std::vector<T>> result;
    result.resize(N, std::vector<T>(M));
    return result;
}

int tentitive_velocity_v(const std::vector<std::vector<double>> vb, //(Nx+1)*(Ny+2)
                         std::vector<std::vector<double>> phi, const std::vector<std::vector<int>> boundary_type,
                         const std::vector<std::vector<double>> boundary_values, double width = 1.0,
                         double height = 1.0, double T = 0.1, double rho = 1.0, double mu = 0.01, int max_iters = 10) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    printf("Nx    : %05d     , Ny     : %05d     , Nt : %05d     , max_iters : %05d.\n", Nx, Ny, Nt, max_iters);
    printf("width : %f, height : %f, T  : %f, rho       : %f.\n", width, height, T, rho);
    printf("dx    : %f, dy     : %f, dt : %f.\n", dx, dy, dt);

    double residual[Nx + 2][Ny + 1] = {{}};

    for (size_t iter = 0; iter < max_iters; iter++) {
        for (size_t i = 1; i < Nx + 1; i++) {
            // 下边界
            if (boundary_type[i][0] == DIRICHLET) { phi[i][0] = boundary_values[i][0]; }
            if (boundary_type[i][0] == NEUMANN) {
                double ghost_phi = phi[i][1] - 2.0 * dy * boundary_values[i][0];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi[i + 1][0] + phi[i - 1][0]) / dx / dx + (phi[i][1] + ghost_phi) / dy / dy;
                phi[i][0]        = (vb[i][0] + mu / rho * p2) / p1;
            }
            // 上边界
            if (boundary_type[i][Ny] == DIRICHLET) { phi[i][Ny] = boundary_values[i][Ny]; }
            if (boundary_type[i][Ny] == NEUMANN) {
                double ghost_phi = phi[i][Ny - 1] + 2.0 * dy * boundary_values[i][Ny];
                double p1        = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2        = (phi[i + 1][Ny] + phi[i - 1][Ny]) / dx / dx + (ghost_phi + phi[i][Ny - 1]) / dy / dy;
                phi[i][Ny]       = (vb[i][Ny] + mu / rho * p2) / p1;
            }
        }

        for (size_t j = 0; j < Ny + 1; j++) {
            // 左边界
            if (boundary_type[0][j] == DIRICHLET) { phi[0][j] = 2 * boundary_values[0][j] - phi[1][j]; }
            if (boundary_type[0][j] == NEUMANN) { phi[0][j] = phi[1][j] - dy * boundary_values[0][j]; }
            // 右边界
            if (boundary_type[Nx + 1][j] == DIRICHLET) { phi[Nx + 1][j] = 2 * boundary_values[Nx + 1][j] - phi[Nx][j]; }
            if (boundary_type[Nx + 1][j] == NEUMANN) { phi[Nx + 1][j] = phi[Nx][j] + dy * boundary_values[Nx + 1][j]; }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny; j++) {
                double p1 = 1 / dt + 2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2 = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                phi[i][j] = (vb[i][j] + mu / rho * p2) / p1;
            }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny; j++) {
                // double p1 = (phi[i - 1][j] - 2 * phi[i][j] + phi[i + 1][j]) /
                // dx / dx; double p2 = (phi[i][j - 1] - 2 * phi[i][j] + phi[i][j
                // + 1]) / dy / dy; residual[i][j] = vb[i][j] - phi[i][j] / dt +
                // mu / rho * (p1 + p2);
                double p1      = +2.0 * mu / rho * (1.0 / dx / dx + 1.0 / dy / dy);
                double p2      = (phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy;
                residual[i][j] = -phi[i][j] * (p1 + 1.0 / dt) + vb[i][j] + (mu / rho * p2);
            }
        }

        // 计算L2误差和L1误差
        double residual_norm_L2 = 0.0;
        double residual_norm_L1 = 0.0;

        for (size_t i = 0; i < Nx + 2; i++) {
            for (size_t j = 0; j < Ny + 1; j++) {
                residual_norm_L1 = std::max(residual_norm_L1, std::abs(residual[i][j]));
                residual_norm_L2 += residual[i][j] * residual[i][j];
            }
        }
        residual_norm_L2 = std::sqrt(residual_norm_L2 * dx * dy);
        printf("iter: %08zu, Error L1 : %.16e, L2 : %.16e.\n", iter, residual_norm_L1, residual_norm_L2);

        // 跳出循环
        if (residual_norm_L2 < EPSILON) break;
    }
}

int main() {
    auto vb              = make_vector_2D<double>(Nx + 2, Ny + 1);
    auto phi             = make_vector_2D<double>(Nx + 2, Ny + 1);
    auto boundary_values = make_vector_2D<double>(Nx + 2, Ny + 1);
    auto boundary_type   = make_vector_2D<int>(Nx + 2, Ny + 1);

    double width  = 1.0;
    double height = 1.0;
    double T      = 0.1;
    double rho    = 1.0;
    double mu     = 0.01;

    double dx        = width / Nx;
    double dy        = height / Ny;
    double dt        = T / Nt;
    int    max_iters = 40;

    // 设置边界条件
    int boundary_edge_type[4] = {1, 2, 2, 2};
    for (size_t i = 1; i < Nx + 1; i++) {
        boundary_type[i][0]  = boundary_edge_type[0];
        boundary_type[i][Ny] = boundary_edge_type[1];
    }
    for (size_t j = 0; j < Ny + 1; j++) {
        boundary_type[0][j]      = boundary_edge_type[2];
        boundary_type[Nx + 1][j] = boundary_edge_type[3];
    }

    // 计算右端项的值
    for (size_t i = 0; i < Nx + 2; i++) {
        for (size_t j = 0; j < Ny + 1; j++) {
            double x = (i - 0.5) * dx;
            double y = j * dy;
            vb[i][j] = (2000.0 * x * y * (x - 1) * (y - 1)
                        - 0.01 * x * (x - 1) * (pow(x, 2) * y * (y - 1) + 2 * x * (2 * y - 1) + 2)
                        - 0.01 * y * (y - 1) * (x * pow(y, 2) * (x - 1) + 2 * y * (2 * x - 1) + 2))
                       * exp(x * y);
        }
    }

    auto fun_u = [](double x, double y) -> double { return x * y * (x - 1) * (y - 1) * exp(x * y); };
    auto fun_f = [](double x, double y) -> double {
        return (2000.0 * x * y * (x - 1) * (y - 1)
                - 0.01 * x * (x - 1) * (pow(x, 2) * y * (y - 1) + 2 * x * (2 * y - 1) + 2)
                - 0.01 * y * (y - 1) * (x * pow(y, 2) * (x - 1) + 2 * y * (2 * x - 1) + 2))
               * exp(x * y);
    };
    auto fun_dudx
        = [](double x, double y) -> double { return y * (y - 1) * (x * y * (x - 1) + 2 * x - 1) * exp(x * y); };
    auto fun_dudy
        = [](double x, double y) -> double { return x * (x - 1) * (x * y * (y - 1) + 2 * y - 1) * exp(x * y); };

    // 计算边界的值
    for (size_t i = 1; i < Nx + 1; i++) {
        if (boundary_type[i][0] == DIRICHLET) { boundary_values[i][0] = fun_u(dx * i - 0.5 * dx, 0); }
        if (boundary_type[i][0] == NEUMANN) { boundary_values[i][0] = fun_dudy(dx * i - 0.5 * dx, 0); }
        if (boundary_type[i][Ny] == DIRICHLET) { boundary_values[i][Ny] = fun_u(dx * i - 0.5 * dx, 1); }
        if (boundary_type[i][Ny] == NEUMANN) { boundary_values[i][Ny] = fun_dudy(dx * i - 0.5 * dx, 1); }
    }

    for (size_t j = 0; j < Ny + 1; j++) {
        if (boundary_type[0][j] == DIRICHLET) { boundary_values[0][j] = fun_u(0, dy * j); }
        if (boundary_type[0][j] == NEUMANN) { boundary_values[0][j] = fun_dudx(0, dy * j); }
        if (boundary_type[Nx + 1][j] == DIRICHLET) { boundary_values[Nx + 1][j] = fun_u(1, dy * j); }
        if (boundary_type[Nx + 1][j] == NEUMANN) { boundary_values[Nx + 1][j] = fun_dudx(1, dy * j); }
    }

    tentitive_velocity_v(vb, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters);
}
