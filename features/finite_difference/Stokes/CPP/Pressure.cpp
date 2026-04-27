
#include <cmath>
#include <iostream>
#include <vector>

// using namespace std;

const int Nx = 200;
const int Ny = 200;
const int Nt = 200;

const int INTERIOR  = 0;
const int DIRICHLET = 1;
const int NEUMANN   = 2;

const double EPSILON = 1e-15;

template <typename T>
auto make_vector_2D(int N, int M) {
    std::vector<std::vector<T>> result;
    result.resize(N, std::vector<T>(M));
    return result;
}

int poisson(std::vector<std::vector<double>> pb, std::vector<std::vector<double>> phi,
            std::vector<std::vector<int>> boundary_type, std::vector<std::vector<double>> boundary_values,
            double width = 1.0, double height = 1.0, double T = 0.1, double rho = 1.0, double mu = 0.01,
            int max_iters = 10) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    printf("Nx    : %d     , Ny     : %d     , Nt : %d     , max_iters : %d.\n", Nx, Ny, Nt, max_iters);
    printf("width : %f, height : %f, T  : %f, rho       : %f.\n", width, height, T, rho);
    printf("dx    : %f, dy     : %f, dt : %f.\n", dx, dy, dt);

    double residual[Nx + 2][Ny + 2] = {{}};

    for (size_t iter = 0; iter < max_iters; iter++) {
        // 处理边界条件
        for (size_t i = 1; i < Nx + 1; i++) {
            // 下边界
            if (boundary_type[i][0] == DIRICHLET) { phi[i][0] = 2 * boundary_values[i][0] - phi[i][1]; }
            if (boundary_type[i][0] == NEUMANN) { phi[i][0] = phi[i][1] - dy * boundary_values[i][0]; }
            // 上边界
            if (boundary_type[i][Ny + 1] == DIRICHLET) { phi[i][Ny + 1] = 2 * boundary_values[i][Ny + 1] - phi[i][Ny]; }
            if (boundary_type[i][Ny + 1] == NEUMANN) { phi[i][Ny + 1] = phi[i][Ny] + dy * boundary_values[i][Ny + 1]; }
        }
        for (size_t j = 1; j < Ny + 1; j++) {
            // 左边界
            if (boundary_type[0][j] == DIRICHLET) { phi[0][j] = 2 * boundary_values[0][j] - phi[1][j]; }
            if (boundary_type[0][j] == NEUMANN) { phi[0][j] = phi[1][j] - dx * boundary_values[0][j]; }
            // 右边界
            if (boundary_type[Nx + 1][j] == DIRICHLET) { phi[Nx + 1][j] = 2 * boundary_values[Nx + 1][j] - phi[Nx][j]; }
            if (boundary_type[Nx + 1][j] == NEUMANN) { phi[Nx + 1][j] = phi[Nx][j] + dx * boundary_values[Nx + 1][j]; }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double a  = 2.0 / dx / dx + 2.0 / dy / dy;
                double b  = ((phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy
                            - pb[i][j]);
                phi[i][j] = b / a;
                // printf(" b/a : %lf, phi[i][j]: %lf.\n", b/a, phi[i][j]);
            }
        }

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
                double a       = 2.0 / dx / dx + 2.0 / dy / dy;
                double b       = ((phi[i + 1][j] + phi[i - 1][j]) / dx / dx + (phi[i][j + 1] + phi[i][j - 1]) / dy / dy
                            - pb[i][j]);
                residual[i][j] = phi[i][j] * a - b;
                // printf("residual[i][j] : %lf.\n", residual[i][j]);
            }
        }

        // 计算L2误差和L1误差
        double residual_norm_L2 = 0.0;
        double residual_norm_L1 = 0.0;

        for (size_t i = 1; i < Nx + 1; i++) {
            for (size_t j = 1; j < Ny + 1; j++) {
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

int poisson(std::vector<std::vector<double>> pb, std::vector<std::vector<double>> phi, int boundary_edge_type[4],
            std::vector<std::vector<double>> boundary_values, double width = 1.0, double height = 1.0, double T = 0.1,
            double rho = 1.0, double mu = 0.01, int max_iters = 10) {
    // 设置边界类型
    auto boundary_type = make_vector_2D<int>(Nx + 2, Ny + 2);
    for (size_t i = 1; i < Nx + 1; i++) {
        boundary_type[i][0]      = boundary_edge_type[0];
        boundary_type[i][Ny + 1] = boundary_edge_type[1];
    }
    for (size_t j = 1; j < Ny + 1; j++) {
        boundary_type[0][j]      = boundary_edge_type[2];
        boundary_type[Nx + 1][j] = boundary_edge_type[3];
    }
    poisson(pb, phi, boundary_type, boundary_values, width, height, T, rho, mu, max_iters);
}

int poisson(const std::vector<std::vector<double>> s,
            const std::vector<std::vector<double>> u, // (Nx+1)*(Ny+2)
            const std::vector<std::vector<double>> v, // (Nx+2)*(Ny+1)
            std::vector<std::vector<double>> phi, int boundary_edge_type[4],
            std::vector<std::vector<double>> boundary_values, double width = 1.0, double height = 1.0, double T = 0.1,
            double rho = 1.0, double mu = 0.01, int max_iters = 10) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    // 设置边界类型
    auto pb = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            pb[i + 1][j + 1]
                = rho / dt * ((u[i + 1][j + 1] - u[i][j + 1]) / dx + (v[i + 1][j + 1] - v[i + 1][j]) / dy - s[i][j]);
        }
    }

    poisson(pb, phi, boundary_edge_type, boundary_values, width, height, T, rho, mu, max_iters);
}

int poisson(const std::vector<std::vector<double>> u, // (Nx+1)*(Ny+2)
            const std::vector<std::vector<double>> v, // (Nx+2)*(Ny+1)
            std::vector<std::vector<double>> phi, int boundary_edge_type[4],
            std::vector<std::vector<double>> boundary_values, double width = 1.0, double height = 1.0, double T = 0.1,
            double rho = 1.0, double mu = 0.01, int max_iters = 10) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dt = T / Nt;

    // 设置边界类型
    auto pb = make_vector_2D<double>(Nx + 2, Ny + 2);
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            pb[i + 1][j + 1] = rho / dt * ((u[i + 1][j + 1] - u[i][j + 1]) / dx + (v[i + 1][j + 1] - v[i + 1][j]) / dy);
        }
    }

    poisson(pb, phi, boundary_edge_type, boundary_values, width, height, T, rho, mu, max_iters);
}

int main() {
    auto pb              = make_vector_2D<double>(Nx + 2, Ny + 2);
    auto phi             = make_vector_2D<double>(Nx + 2, Ny + 2);
    auto s               = make_vector_2D<double>(Nx + 2, Ny + 2);
    auto boundary_values = make_vector_2D<double>(Nx + 2, Ny + 2);

    double width  = 1.0;
    double height = 1.0;
    double T      = 0.1;
    double rho    = 1.0;
    double mu     = 0.01;

    double dx        = width / Nx;
    double dy        = height / Ny;
    double dt        = T / Nt;
    int    max_iters = 400;

    // 设置右端项
    for (size_t i = 1; i < Nx + 1; i++) {
        for (size_t j = 1; j < Ny + 1; j++) {
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            pb[i][j] = (x * (x - 1) * (std::pow(x, 2) * y * (y - 1) + 2 * x * (2 * y - 1) + 2)
                        + y * (y - 1) * (x * std::pow(y, 2) * (x - 1) + 2 * y * (2 * x - 1) + 2))
                       * std::exp(x * y);
        }
    }

    // 设置边界类型
    int boundary_edge_type[4] = {1, 1, 1, 1};

    poisson(pb, phi, boundary_edge_type, boundary_values, width, height, T, rho, mu, max_iters);
}
