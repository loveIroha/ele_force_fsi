/// @date 2023-06-30
/// @file algebra.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 代数运算函数
///
///

#ifndef __ALGEBRA_H__
#define __ALGEBRA_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace algebra {

struct double2 {
    double x, y;
};

// 赋值
template <typename TV>
void assign(TV& a, TV b) {
    a = b;
}

// 内积
int    inner(int a, int b) { return a * b; }
double inner(double a, double b) { return a * b; }
double inner(double2 a, double2 b) { return (a.x * b.x) + (a.y * b.y); }

// 归零
template <typename TV>
void zero(TV& a) {
    a = TV{};
}

// 线性运算
template <typename T, typename TV>
void axpy(T a, const TV& x, TV& y) {
    y = y + a * x;
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

template <typename TV, typename T = double>
T inner(std::vector<TV> a, std::vector<TV> b) {
    // LOG_SCOPE_FUNCTION(INFO);
    assert(a.size() == b.size());
    T sum{};
    for (size_t i = 0; i < a.size(); i++) {
        sum += inner(a[i], b[i]);
        // LOG_F(INFO, "内积 :  %.16e .", sum);
    }
    return sum;
}

template <typename TV, typename T = double>
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

template <typename TV, typename T = double>
T distance(const std::vector<TV>& a, const std::vector<TV>& b) {
    std::vector<TV> c = a;
    // c = -b + c
    axpy(-1.0, b, c);
    return norm<TV, T>(c);
}

class NonlinearProblem {
  public:
    std::vector<double> xk;
    std::vector<double> rk;
    void                apply(const std::vector<double>& x, std::vector<double>& y) {
        double epsilon_max     = 1e5;
        double epsilon_machine = std::numeric_limits<double>::epsilon();
        double epsilon         = std::min(
            std::sqrt((1.0 + norm<double, double>(xk)) * epsilon_machine) / norm<double, double>(x), epsilon_max);

        // LOG_F(INFO, "epsilon:  %.16e", epsilon);

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

    // virtual void h(const std::vector<double> &x, std::vector<double> &y)
    // const = 0;
    virtual void h(const std::vector<double>& x, std::vector<double>& y) = 0;
};

template <typename Problem, typename VectorType>
auto bicgstab(VectorType& x, const VectorType& rhs, Problem& problem, size_t size, size_t max_iteration = 1000,
              double tol_i = 1e-6, double tol = 1e-9, bool silent = false) {
    // LOG_SCOPE_FUNCTION(INFO);
    // LOG_F(INFO, "启动bicgstab求解器");

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
    // LOG_F(INFO, "tol %.16e,tol_i %.16e,rhs_sqnorm %.16e,r0_sqnorm %.16e .",
    // tol, tol_i, rhs_sqnorm, r0_sqnorm);

    double eps2 = std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();

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
        // LOG_F(INFO, "第%d次迭代，残差为 %.16e", i, norm<double, double>(r));
        // LOG_F(INFO, "tol2: %.16e, tol2_i: %.16e.", tol2, tol2_i);

        double rho_old = rho;
        rho            = inner<double, double>(r0, r);
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
        // LOG_F(INFO, "beta");
        if (!std::isnormal(beta)) break;

        // p = r + beta * (p - w * v);
        axpy(-w, v, p);
        axpy(beta, p, r, p);

        // y = precond.solve(p);
        y = p;

        problem.apply(y, v);
        alpha = rho / inner<double, double>(r0, v);
        // LOG_F(INFO, "alpha");
        if (!std::isnormal(alpha)) break;
        // s = r - alpha * v;
        axpy(-alpha, v, r, s);

        // z = precond.solve(s);
        z = s;

        problem.apply(z, t);

        // 代码参考了eigen3，但是修改了中断的处理方式
        w = inner<double, double>(t, s) / inner<double, double>(t, t);
        // LOG_F(INFO, "w");
        if (!std::isnormal(w)) break;

        axpy(alpha, y, x);
        axpy(w, z, x);
        axpy(-w, t, s, r);
        // x += alpha * y + w * z;
        // r = s - w * t;
        ++i;
        // LOG_F(INFO, "%d", i);
    }

    double tol_error = std::sqrt(inner<double, double>(r, r) / rhs_sqnorm);
    bool   success   = tol < tol_error ? false : true;
    // LOG_F(INFO, "成功：%d，总迭代次数为%d，残差为 %.16e，相对误差为 %.16e",
    // success, i, norm<double, double>(r), tol_error);
    return std::make_tuple(success, tol_error, i);
}

template <typename Problem, typename VectorType>
auto newton_raphson(VectorType& x0, Problem& problem, size_t linear_max_iteration = 100, double linear_tol_i = 1e-3,
                    double linear_tol = 1e-6, double nonlinear_tol = 1e-4, double nonlinear_max_iteration = 100,
                    bool silent = false) {
    // LOG_SCOPE_FUNCTION(INFO);
    // LOG_F(INFO, "启动newton_raphson求解器");

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

    if (algebra::inner<double, double>(rk, rk) < eps2) return std::make_tuple(true, 0.0, 0);

    do {
        i++;
        dx = rk;
        auto [success, tol_error, i]
            = bicgstab(dx, rk, problem, n, linear_max_iteration, linear_tol_i, linear_tol, silent);

        axpy(1.0, dx, xk);
        problem.h(xk, rk);
        scale(-1.0, rk);
        auto a = inner<double, double>(rk, rk);
        auto b = inner<double, double>(xk, xk);
        ek     = std::sqrt(a / b);
        // LOG_F(INFO, "Relative residual : %e, Absolute residual : %e.", ek,
        // std::sqrt(a));
        if (a < eps2) break;
        if (i > nonlinear_max_iteration) break;
    } while (ek > nonlinear_tol);

    x0 = xk;
    bool success;
    success = ek < nonlinear_tol ? false : true;
    // LOG_F(INFO, "Relative residual : %e.", ek);
    return std::make_tuple(success, ek, i);
}

template <typename TV, typename T>
std::vector<T> flatten(const std::vector<TV>& v) {
    constexpr int       TV_size = sizeof(TV) / sizeof(T);
    std::vector<double> s(v.size() * TV_size);
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < TV_size; j++) {
            s[TV_size * i + j] = ((T*)&v[i])[j];
        }
    }
    return s;
}

template <typename TV, typename T>
std::vector<TV> ripple(const std::vector<T>& s) {
    constexpr int TV_size = sizeof(TV) / sizeof(T);
    assert(s.size() % TV_size == 0);

    std::vector<TV> v(s.size() / TV_size);
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < TV_size; j++) {
            ((T*)&v[i])[j] = s[TV_size * i + j];
        }
    }
    return v;
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& input) {
    assert(input.size() > 0);

    size_t Nx = input.size();
    size_t Ny = input[0].size();

    std::vector<T> result(Nx * Ny);
    for (size_t i = 0; i < Nx; i++) {
        assert(input[i].size() == Ny);
        for (size_t j = 0; j < Ny; j++) {
            result[i + j * Nx] = input[i][j];
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> ripple(const std::vector<T>& input, size_t Nx) {
    assert(input.size() % Nx == 0);
    size_t                      Ny = input.size() / Nx;
    std::vector<std::vector<T>> result(Nx);

    for (size_t i = 0; i < Nx; i++) {
        result[i].resize(Ny);
        for (size_t j = 0; j < Ny; j++) {
            result[i][j] = input[i + j * Nx];
        }
    }
    return result;
}

} // namespace algebra

#endif // __ALGEBRA_H__