/// @date 2023-04-12
/// @file test_lid_driven_flow.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试方腔驱动流
///
///

#include <PhysicsSolver/LidDrivenFlow/LidDrivenFlow.h>

int main() {
    int           n   = 257;
    double        dt  = 0.01;
    double        mu  = 0.01;
    double        rho = 1.0;
    int3          dim = make_int3(n, n, n);
    double3       dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};
    LidDrivenFlow flow(dim, dh, dt, mu, rho);

    std::vector<double3> fn(n * n * n);
    std::vector<double3> un(n * n * n);
    std::vector<double3> u(n * n * n);
    std::vector<double>  p(n * n * n);
    flow.u0.get(un);

    for (int i = 0; i < 10; i++) {
        flow.solveOneStep(fn, un, u, p);
        for (int j = 0; j < n * n * n; j++) {
            un[j] = u[j];
        }
    }

    LOG_F(WARNING, "Velocity at the center: %.16e, %.16e, %.16e.", un[n * n * n / 2].x, un[n * n * n / 2].y,
          un[n * n * n / 2].z);
    LOG_F(WARNING, "Pressure at the center: %.16e.", p[n * n * n / 2]);

    // results at the 100th step
    // Velocity at the center: -1.0622815859085939e-01, -2.1455341880285850e-03,
    // -1.1170269367949218e-04. Pressure at the center: 4.3534405654139408e-02.

    // results at the 10th step
    // Velocity at the center: -1.9748945943839559e-02, -1.5244704546451486e-03,
    // -7.8727562693345525e-05. Pressure at the center: 4.6871490851597232e-02.

    // Velocity at the center: -1.9748945943839556e-02, -7.8727562693354009e-05,
    // -1.5244704546451605e-03. Pressure at the center: 4.6871490851597065e-02.
    // flow.record();

    return true;
}
