#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace algebra {

// 赋值
template <typename TV>
void assign(TV& a, TV b) {
    a = b;
}

// 内积
int    inner(int a, int b) { return a * b; }
double inner(double a, double b) { return a * b; }

// 归零
template <typename TV>
void zero(TV& a) {
    a = TV{};
}

// 线性运算
template <typename T, typename TV>
void axpy(T a, const TV& x, TV& y) {
    y += a * x;
}

template <typename T, typename TV>
void axpy(T a, const TV& x, const TV& y, TV& z) {
    z = y + a * x;
}

template <typename TV>
void assign(std::vector<TV>& a, std::vector<TV> b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++) {
        assign(a[i], b[i]);
    }
}

template <typename TV, typename T>
T inner(std::vector<TV> a, std::vector<TV> b) {
    assert(a.size() == b.size());
    T sum{};
    for (size_t i = 0; i < a.size(); i++) {
        sum += inner(a[i], b[i]);
    }
    return sum;
}

template <typename TV, typename T>
T norm(std::vector<TV> a) {
    return std::sqrt(inner<TV, T>(a, a));
}

template <typename TV>
void zero(std::vector<TV>& a) {
    for (size_t i = 0; i < a.size(); i++) {
        zero(a[i]);
    }
}

template <typename T, typename TV>
void axpy(T a, const std::vector<TV>& x, std::vector<TV>& y) {
    assert(x.size() == y.size());
    for (size_t i = 0; i < x.size(); i++) {
        axpy(a, x[i], y[i]);
    }
}

template <typename T, typename TV>
void scale(T a, TV& x) {
    x *= a;
}

template <typename T, typename TV>
void scale(T a, std::vector<TV>& x) {
    for (size_t i = 0; i < x.size(); i++) {
        scale(a, x[i]);
    }
}

template <typename T, typename TV>
void axpy(T a, const std::vector<TV>& x, const std::vector<TV>& y, std::vector<TV>& z) {
    assert(x.size() == y.size());
    assert(x.size() == z.size());
    for (size_t i = 0; i < x.size(); i++) {
        axpy(a, x[i], y[i], z[i]);
    }
}

} // namespace algebra

using namespace algebra;

class Problem {
  public:
    void apply(const std::vector<double>& x, std::vector<double>& y) const {
        y[0] = 1.0 * x[0];
        y[1] = 2.0 * x[1];
        y[2] = 3.0 * x[2];
        y[3] = 4.0 * x[3];
    }
};

class NonlinearProblem {
  public:
    std::vector<double> xk;
    std::vector<double> rk;
    void                apply(const std::vector<double>& x, std::vector<double>& y) const {
        double epsilon_machine = std::numeric_limits<double>::epsilon();
        double epsilon = std::sqrt((1.0 + norm<double, double>(xk)) * epsilon_machine) / norm<double, double>(x);

        assert(x.size() == y.size());
        assert(x.size() == xk.size());
        assert(x.size() == rk.size());

        std::vector<double> a(x.size());
        std::vector<double> b(x.size());

        // a = xk + epsilon*x
        axpy(epsilon, x, xk, a);
        // b = h(a)
        h(a, b);
        // y = (h(a)-h(xk))/epsilon;
        // y = (h(a)+rk)/epsilon;
        // y = (b+rk)/epsilon;
        zero(y);
        axpy(1.0 / epsilon, b, y);
        axpy(1.0 / epsilon, rk, y);
    }

    void h(const std::vector<double>& x, std::vector<double>& y) const {
        y[0] = 1.0 * x[0];
        y[1] = 2.0 * x[1];
        y[2] = 3.0 * x[2];
        y[3] = 4.0 * x[3];
    }
};

template <typename Problem, typename VectorType>
auto bicgstab(VectorType& x, const VectorType& rhs, const Problem& problem, size_t size, size_t max_iteration = 1000,
              double tol_i = 1e-6, double tol = 1e-9, bool silent = false) {
    auto       n = size;
    VectorType r(n), r0(n), Ax(n);

    // calculate Ax
    // r = rhs - Ax
    problem.apply(x, Ax);
    axpy(-1, Ax, rhs, r);
    assign(r0, r);

    double r0_sqnorm  = inner<double, double>(r0, r0);
    double rhs_sqnorm = inner<double, double>(rhs, rhs);

    double tol2   = tol * tol * rhs_sqnorm;
    double tol2_i = tol_i * tol_i * r0_sqnorm;
    double eps2   = std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();

    // NOTE: 1. 判断右端项是否为零，若为零，则解为零。
    if (rhs_sqnorm < eps2) {
        zero(x);
        return std::make_tuple(true, 0.0, 0);
    }
    double rho   = 1;
    double alpha = 1;
    double w     = 1;

    VectorType v(n), p(n);
    VectorType y(n), z(n);
    VectorType kt(n), ks(n);
    VectorType s(n), t(n);

    int i        = 0;
    int restarts = 0;

    while (inner<double, double>(r, r) > tol2 && inner<double, double>(r, r) > tol2_i && i < max_iteration) {
        auto temp_4 = inner<double, double>(r, r);

        double rho_old = rho;

        rho = inner<double, double>(r0, r);
        if (std::abs(rho) < eps2) {
            // The new residual vector became too orthogonal to the arbitrarily
            // chosen direction r0 Let's restart with a new r0:
            problem.apply(x, Ax);
            axpy(-1, Ax, rhs, r);
            assign(r0, r);
            rho = r0_sqnorm = inner<double, double>(r, r);
            if (restarts++ == 0) i = 0;
        }
        double beta = (rho / rho_old) * (alpha / w);
        if (!std::isnormal(beta)) break;

        // p = r + beta * (p - w * v);
        axpy(-w, v, p);
        axpy(beta, p, r, p);

        // y = precond.solve(p);
        y = p;

        problem.apply(y, v);
        alpha = rho / inner<double, double>(r0, v);
        if (!std::isnormal(alpha)) break;
        // s = r - alpha * v;
        axpy(-alpha, v, r, s);

        // z = precond.solve(s);
        z = s;

        problem.apply(z, t);

        // 代码参考了eigen3，但是修改了中断的处理方式
        w = inner<double, double>(t, s) / inner<double, double>(t, t);
        if (!std::isnormal(w)) break;

        axpy(alpha, y, x);
        axpy(w, z, x);
        axpy(-w, t, s, r);
        // x += alpha * y + w * z;
        // r = s - w * t;
        ++i;
    }

    double tol_error = std::sqrt(inner<double, double>(r, r) / rhs_sqnorm);
    bool   success;
    success = tol < tol_error ? false : true;
    return std::make_tuple(success, tol_error, i);
}

template <typename Problem, typename VectorType>
auto newton_raphson(VectorType& x0, Problem& problem, size_t linear_max_iteration = 100, double linear_tol_i = 1e-6,
                    double linear_tol = 1e-9, double nonlinear_tol = 1e-8, double nonlinear_max_iteration = 100,
                    bool silent = false) {
    int    n = x0.size();
    int    i = 0;
    double ek;
    double eps2 = std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();

    VectorType  dx(n);
    VectorType& xk = problem.xk;
    VectorType& rk = problem.rk;

    xk = x0;
    problem.h(xk, rk);
    scale(-1.0, rk);

    do {
        i++;
        dx = rk;
        bicgstab(dx, rk, problem, n, linear_max_iteration, linear_tol_i, linear_tol, silent);
        axpy(1.0, dx, xk);
        problem.h(xk, rk);
        scale(-1.0, rk);
        auto a = inner<double, double>(rk, rk);
        auto b = inner<double, double>(xk, xk);
        ek     = std::sqrt(a / b);
        printf("ek = %e, %e\n", ek, std::sqrt(a));
        if (a < eps2) break;
        if (i > nonlinear_max_iteration) break;
    } while (ek > nonlinear_tol);

    x0 = xk;
    bool success;
    success = ek < nonlinear_tol ? false : true;
    printf("ek = %e\n", ek);
    return std::make_tuple(success, ek, i);
}

int main() {
    std::vector<double> a = {1, 2, 3, 4};
    std::vector<double> b = {1, 2, 3, 4};
    std::vector<double> c = {1, 2, 3, 4};
    std::vector<double> d = {1, 2, 3, 4};
    // Problem problem;

    // // 测试简单的线性运算
    // axpy(1, b, a);
    // axpy(1, b, c, d);
    // std::cout << inner<double, double>(a, a) << std::endl;
    // std::cout << inner<double, double>(d,d) << std::endl;

    // // 测试矩阵向量乘法
    // problem.apply(a, b);

    // 测试bicgstab
    // bicgstab(a, b, problem, 4);

    NonlinearProblem problem;
    problem.xk.resize(4);
    problem.rk.resize(4);
    newton_raphson(a, problem);
    std::cout << inner<double, double>(a, a) << std::endl;

    return 0;
}
