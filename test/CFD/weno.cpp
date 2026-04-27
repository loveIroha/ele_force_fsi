#include <iostream>
#include <limits>
namespace cfd {
void upwind() {}

/// @brief dissipative term. 耗散项。
///        Different approximative Riemann solvers use different definitions
///        for dissipative function.
template <typename T>
T dissipative_term(T u_l, T u_r) {
    return std::abs(u_l - u_r);
}

/// @brief upwind-type numerical flux. 迎风类型的数值通量。
///        A special choice for the approximate Riemann solver.
/// @param fun conserved quantity (守恒量如动量 $\rho u$ 等等)
/// @param dis disspative term
template <typename T, typename Fun, typename Dis>
T upwind_flux(T u_l, T u_r, Fun fun, Dis dis) {
    double f_l = fun(u_l);
    double f_r = fun(u_r);
    return 0.5 * (fun(u_l) + fun(u_r) - dis(f_l, f_r));
}

/// @brief use MUSCL scheme to reconstruct u_l and u_r
template <typename T>
T muscl_left(T u_0, T u_1, T u_2, T u_3, T kappa = 1.0) {
    return u_1 + 0.25 * (1.0 + kappa) * (u_2 - u_1) + 0.25 * (1.0 - kappa) * (u_1 - u_0);
}

template <typename T>
T muscl_right(T u_0, T u_1, T u_2, T u_3, T kappa = 1.0) {
    return muscl_left(u_1, u_2, u_3, 0.0, kappa);
    // return u_2 + 0.25 * (1.0 + kappa) * (u_1 - u_2) + 0.25 * (1.0 - kappa) *
    // (u_2 - u_3);
}

/// @brief use WENO5 scheme to reconstruct u_l and u_r
template <typename T>
T weno_5th_left(T u_0, T u_1, T u_2, T u_3, T u_4, T u_5) {
    double p0 = 1.0 / 3.0 * u_0 - 7.0 / 6.0 * u_1 + 11.0 / 6.0 * u_2;
    double p1 = -1.0 / 6.0 * u_1 + 5.0 / 6.0 * u_2 + 1.0 / 3.0 * u_3;
    double p2 = 1.0 / 3.0 * u_2 + 5.0 / 6.0 * u_3 - 1.0 / 6.0 * u_4;
    printf("p0 : %f, p1 : %f, p2 : %f.\n", p0, p1, p2);

    double beta_0 = 13.0 / 12.0 * (u_0 - 2.0 * u_1 + u_2) * (u_0 - 2.0 * u_1 + u_2)
                    + 1.0 / 4.0 * (u_0 - 4.0 * u_1 + 3.0 * u_2) * (u_0 - 4.0 * u_1 + 3.0 * u_2);
    double beta_1
        = 13.0 / 12.0 * (u_1 - 2.0 * u_2 + u_3) * (u_1 - 2.0 * u_2 + u_3) + 1.0 / 4.0 * (u_1 - u_3) * (u_1 - u_3);
    double beta_2 = 13.0 / 12.0 * (u_2 - 2.0 * u_3 + u_4) * (u_2 - 2.0 * u_3 + u_4)
                    + 1.0 / 4.0 * (3.0 * u_2 - 4.0 * u_3 + u_4) * (3.0 * u_2 - 4.0 * u_3 + u_4);
    printf("beta_0 : %f, beta_1 : %f, beta_2 : %f.\n", beta_0, beta_1, beta_2);

    // double epsilon = 1e-6;
    double epsilon = std::numeric_limits<double>::epsilon();
    double gamma_0 = 0.1, gamma_1 = 0.6, gamma_2 = 0.3;

    double alpha_0 = gamma_0 / (beta_0 + epsilon) / (beta_0 + epsilon);
    double alpha_1 = gamma_1 / (beta_1 + epsilon) / (beta_1 + epsilon);
    double alpha_2 = gamma_2 / (beta_2 + epsilon) / (beta_2 + epsilon);

    printf("alpha_0 : %f, alpha_1 : %f, alpha_2 : %f.\n", alpha_0, alpha_1, alpha_2);

    double omega_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
    double omega_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2);
    double omega_2 = alpha_2 / (alpha_0 + alpha_1 + alpha_2);

    printf("omega_0 : %f, omega_1 : %f, omega_2 : %f.\n", omega_0, omega_1, omega_2);

    return omega_0 * p0 + omega_1 * p1 + omega_2 * p2;
}

template <typename T>
T weno_5th_right(T u_0, T u_1, T u_2, T u_3, T u_4, T u_5) {
    return weno_5th_left(u_5, u_4, u_3, u_2, u_1, u_0);
}

} // namespace cfd

template <typename T>
class Momentum {
  public:
    double v = 1.0;

    void interpolate(T u_0, T u_1, T u_2, T u_3) { v = -0.0625 * u_0 + 0.5625 * u_1 + 0.5625 * u_2 - 0.0625 * u_3; }
    T    operator()(T u) {
        printf("u : %f\n", u);
        return u * u + u * v;
    }
};

int main() {
    cfd::upwind();
    double a = cfd::weno_5th_left(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    double c = cfd::weno_5th_left(1.0, 2.0, 2.0, 4.0, 4.0, 6.0);
    double b = cfd::weno_5th_right(1.0, 2.0, 2.0, 4.0, 4.0, 6.0);
    printf("%.16e\n", a);
    printf("%f\n", c);
    printf("%f\n", b);

    // calculate flux of momentum at cell boundary
    //              |i-1| i |i+1| i+2| i+3| i+4| i+5| i+6|
    //              |
    // ---*---*---*---*---*---*--*--*--*--*--*--*--*--*--
    Momentum<double> momentum;
    momentum.interpolate(1.0, 2.0, 3.0, 4.0);
    double u_l   = cfd::weno_5th_left(1.0, 2.0, 2.0, 4.0, 4.0, 6.0);
    double u_r   = cfd::weno_5th_right(1.0, 2.0, 2.0, 4.0, 4.0, 6.0);
    double f_hat = cfd::upwind_flux(u_l, u_r, momentum, cfd::dissipative_term<double>);
    printf("%f\n", f_hat);
    return 0;
}