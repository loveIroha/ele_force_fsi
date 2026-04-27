#include <AlgebraSolver/algebra.h>
#include <PhysicsSolver/StokesFlow2D/MonolithicSolver.h>
#include <io/vector_io.h>

#include <iostream>

int main() {
    size_t              Nx = 10;
    size_t              Ny = 10;
    double              hx = 1.0;
    double              hy = 1.0;
    NavierStokesLocal2D nsl_2D(0.1, 0.2);
    auto                u = IO::make_vector_2D<double>(Nx + 3, Ny + 2);
    auto                v = IO::make_vector_2D<double>(Nx + 2, Ny + 3);
    auto                p = IO::make_vector_2D<double>(Nx + 2, Ny + 2);

    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            double x[5] = {u[i + 1][j + 1], u[i + 2][j + 1], v[i + 1][j + 1], v[i + 1][j + 2], p[i + 1][j + 1]};
            double b[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            b[0]        = -u[i][j + 1] / hx / hx - u[i + 1][j] / hy / hy - u[i + 1][j + 2] / hy / hy - p[i][j + 1] / hx;
            b[1]
                = -u[i + 3][j + 1] / hx / hx - u[i + 2][j] / hy / hy - u[i + 2][j + 2] / hy / hy + p[i + 2][j + 1] / hx;
            b[1] = -v[i][j + 1] / hx / hx - v[i + 2][j + 1] / hx / hx - v[i + 1][j + 2] / hy / hy - p[i + 1][j] / hy;
            b[1]
                = -v[i][j + 2] / hx / hx - v[i + 2][j + 2] / hx / hx - v[i + 1][j + 3] / hy / hy + p[i + 1][j + 2] / hy;

            nsl_2D.solve(x, b);
            u[i + 1][j + 1] = x[0];
            u[i + 2][j + 1] = x[1];
            v[i + 1][j + 1] = x[2];
            v[i + 1][j + 2] = x[3];
            p[i + 1][j + 1] = x[4];
        }
    }

    std::cout << "原始矩阵：" << std::endl << nsl_2D.matrix << std::endl;
    std::cout << "逆矩阵：" << std::endl << nsl_2D.inverse << std::endl;

    std::cout << nsl_2D(-10, -10) << std::endl;
    std::cout << nsl_2D(2, 3) << std::endl;

    // nsl_2D.solve(x, b);

    // for (size_t i = 0; i < 5; i++)
    // {
    //     std::cout << x[i] << std::endl;
    //     /* code */
    // }

    return 0;
}
