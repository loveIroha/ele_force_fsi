#include <boost/math/tools/roots.hpp>

#include <iostream>

#define NPUHEART_EPS 1e-6
int main() {
    double x[3] = {9, -0.1, -0.1};
    double values[3];

    // constants
    const double r_short_endo = 7;
    const double r_short_epi  = 10;
    const double r_long_endo  = 17;
    const double r_long_epi   = 20;

    // 求解透壁坐标需要求解四阶方程，用二分法求解
    auto fun = [&](double t) {
        double rs  = r_short_endo + (r_short_epi - r_short_endo) * t;
        double rl  = r_long_endo + (r_long_epi - r_long_endo) * t;
        double a2  = x[1] * x[1] + x[2] * x[2];
        double b2  = x[0] * x[0];
        double rs2 = rs * rs;
        double rl2 = rl * rl;
        double drs = (r_short_epi - r_short_endo) * t;
        double drl = (r_long_epi - r_long_endo) * t;

        double f  = a2 * rl2 + b2 * rs2 - rs2 * rl2;
        double df = 2.0 * (a2 * rl * drl + b2 * rs * drs - rs * drs * rl2 - rs2 * rl * drl);
        std::cout << t << "  " << f << "  " << df << std::endl;

        return boost::math::make_tuple(f, df);
    };

    int    digits = std::numeric_limits<double>::digits;
    double t      = boost::math::tools::newton_raphson_iterate(fun, 0.5, 0.0, 1.0, 0.000001);
    values[0]     = t;

    double r_short = r_short_endo * (1 - t) + r_short_epi * t;
    double r_long  = r_long_endo * (1 - t) + r_long_epi * t;

    double a = std::sqrt(x[1] * x[1] + x[2] * x[2]) / r_short;
    double b = x[0] / r_long;

    // mu
    values[1] = std::atan2(a, b);

    // theta
    values[2] = (values[1] < NPUHEART_EPS) ? 0.0 : M_PI - std::atan2(x[2], -x[1]);

    std::cout << values[0] << "  " << values[1] << "  " << values[2] << std::endl;
}
