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

#include <io.h>
#include <vector_types.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>
namespace algebra {
enum class Norm { l1, l2, linf };

// 赋值
template <typename TV>
void assign(TV& a, TV b) {
    a = b;
}

// 内积
int    inner(int a, int b) { return a * b; }
double inner(double a, double b) { return a * b; }
double inner(double2 a, double2 b) { return (a.x * b.x) + (a.y * b.y); }
double inner(double3 a, double3 b) { return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); }

// 求和
double sum(double a) { return a; }
double sum(double2 a) { return a.x + a.y; }
double sum(double3 a) { return a.x + a.y + a.z; }

// 归零
template <typename TV>
void zero(TV& a) {
    a = TV{};
}

template <typename TV>
void zero(std::vector<TV>& a) {
    for (size_t i = 0; i < a.size(); i++) {
        zero(a[i]);
    }
}

// 边界归零
void zero_boundary(double& x) {}

template <typename TV>
void zero_boundary(std::vector<TV>& x) {
    // for (size_t i = 0; i < x.size(); i++)
    // {
    //     for (size_t j = 0; j < x[i].size(); j++)
    //     {
    //         if (i == 0 || i == x.size() - 1 || j == 0 || j == x[i].size() -
    //         1)
    //         {
    //             x[i][j] = TV{};
    //         }
    //     }
    // }

    zero(x[0]);
    zero(x[x.size() - 1]);
    for (size_t i = 1; i < x.size() - 1; i++) {
        zero_boundary(x[i]);
    }
}

void add(double& x, double a) { x += a; }
void add(double2& x, double a) {
    x.x += a;
    x.y += a;
}

template <typename T, typename TV>
void add(std::vector<TV>& x, T a) {
    for (size_t i = 0; i < x.size(); i++) {
        add(x[i], a);
    }
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

// 可嵌套
template <typename TV, typename T = double>
T inner(std::vector<TV> a, std::vector<TV> b) {
    assert(a.size() == b.size());
    T result{};
    for (size_t i = 0; i < a.size(); i++) {
        result += inner(a[i], b[i]);
    }
    return result;
}

int size(double a) { return 1; }
int size(double2 a) { return 2; }
template <typename TV>
int size(std::vector<TV> a) {
    int result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result += size(a[i]);
    }
    return result;
}

// 可嵌套模板
template <typename TV, typename T = double>
T sum(std::vector<TV> a) {
    T result{};
    for (size_t i = 0; i < a.size(); i++) {
        result += sum(a[i]);
    }
    return result;
}

template <typename TV>
double average(std::vector<TV> a) {
    return sum(a) / size(a);
}

double  abs(double a) { return std::abs(a); }
double2 abs(double2 a) { return {std::abs(a.x), std::abs(a.y)}; }
double3 abs(double3 a) { return {std::abs(a.x), std::abs(a.y), std::abs(a.z)}; }

template <typename TV>
std::vector<TV> abs(const std::vector<TV>& a) {
    auto b = a;
    for (size_t i = 0; i < a.size(); i++) {
        b[i] = abs(a[i]);
    }
    return b;
}

// double max(double a, double b) { return std::max(a, b); }
// double max(double2 a, double2 b) { return a.x; }
double max(double a) { return a; }
double max(double2 a) { return std::max(a.x, a.y); }
double max(double3 a) { return std::max(std::max(a.x, a.y), a.z); }

// template <typename TV>
// double max(const std::vector<TV> &a, double b)
// {
//     return max(max(a), b);
// }

template <typename TV>
double max(const std::vector<TV>& a) {
    double b = std::numeric_limits<double>::min();
    for (size_t i = 0; i < a.size(); i++) {
        auto c = a[i];
        b      = std::max(max(c), b);
    }
    return b;
}

template <typename TV, typename T = double>
T norm(const std::vector<TV>& a, Norm norm_type = Norm::l2) {
    switch (norm_type) {
    case Norm::l1:
        return sum(abs(a));
    case Norm::l2:
        return std::sqrt(inner<TV, T>(a, a));
    case Norm::linf:
        return max(abs(a));
    default:
        CHECK_F(false, "Not implemented for this norm type.");
        return 0.0;
    }
}

template <typename TV, typename T = double>
T norm(const TV& a, Norm norm_type = Norm::l2) {
    switch (norm_type) {
    case Norm::l1:
        return sum(abs(a));
    case Norm::l2:
        return std::sqrt(inner(a, a));
    case Norm::linf:
        return max(abs(a));
    default:
        CHECK_F(false, "Not implemented for this norm type.");
        return 0.0;
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
    x = a * x;
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

template <typename TV, typename T = double>
T distance(const TV& a, const TV& b) {
    TV c = a - b;
    // c = -b + c
    // axpy(-1.0, b, c);
    return norm(c);
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

        LOG_F(8, "计算得 epsilon:  %.16e", epsilon);

        assert(x.size() == y.size());
        assert(x.size() == xk.size());
        assert(x.size() == rk.size());

        std::vector<double> a(x.size());
        std::vector<double> b(x.size());

        LOG_F(8, "Objective function $\\|\\mathcal{xk}\\|_2$ = %.16e", algebra::norm<double, double>(xk));
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
    ScopeProfiler _{__func__};
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "启动bicgstab求解器");

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
    LOG_F(INFO, "右端项的范数      %.16e.", rhs_sqnorm);
    LOG_F(INFO, "残差的范数        %.16e.", r0_sqnorm);
    LOG_F(INFO, "绝对误差阈值      %.16e.", tol);
    LOG_F(INFO, "相对误差阈值      %.16e.", tol_i);
    LOG_F(INFO, "相对于右端项的误差 %.16e.", tol2);
    LOG_F(INFO, "相对于残差的误差   %.16e.", tol2_i);

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
        LOG_F(INFO, "第%d次迭代，残差为 %.16e", i, norm<double, double>(r));

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
        LOG_F(INFO, "第 %d 次BICGSTAB迭代", i);
    }

    double tol_error = std::sqrt(inner<double, double>(r, r) / rhs_sqnorm);
    bool   success   = tol < tol_error ? false : true;
    LOG_F(INFO, "成功：%d，总迭代次数为%d，残差为 %.16e，相对误差为 %.16e", success, i, norm<double, double>(r),
          tol_error);
    return std::make_tuple(success, tol_error, i);
}

template <typename Problem, typename VectorType>
auto newton_raphson(VectorType& x0, Problem& problem, size_t linear_max_iteration = 100, double linear_tol_i = 1e-3,
                    double linear_tol = 1e-6, double nonlinear_tol = 1e-4, double nonlinear_max_iteration = 100,
                    bool silent = false) {
    ScopeProfiler _{__func__};
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "启动newton_raphson求解器");

    int    n = x0.size();
    int    i = 0;
    double ek;
    double eps2 = std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();

    VectorType  dx(n);
    VectorType& xk = problem.xk;
    VectorType& rk = problem.rk;

    xk = x0;
    rk.resize(xk.size());

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
        LOG_F(INFO, "Relative residual : %e, Absolute residual : %e.", ek, std::sqrt(a));
        if (a < eps2) break;
        if (i > nonlinear_max_iteration) break;
    } while (ek > nonlinear_tol);

    x0 = xk;
    bool success;
    success = ek < nonlinear_tol ? false : true;
    LOG_F(INFO, "Relative residual : %e.", ek);
    return std::make_tuple(success, ek, i);
}

// 返回三维数据的一个截面
// TODO : 实现x方向和y方向的函数
template <typename T>
std::vector<T> slice(const std::vector<T>& input, size_t Nx, size_t Ny, size_t Nz, size_t z_index) {
    CHECK_F(input.size() == Nx * Ny * Nz);
    CHECK_F(z_index < Nz);

    std::vector<T> result(Nx * Ny);
    for (size_t i = 0; i < Nx; i++) {
        for (size_t j = 0; j < Ny; j++) {
            result[i + j * Nx] = input[i + j * Nx + z_index * Nx * Ny];
        }
    }
    return result;
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
std::vector<T> flatten(const std::vector<std::vector<std::vector<T>>>& input) {
    assert(input.size() > 0);

    size_t Nx = input.size();
    size_t Ny = input[0].size();
    size_t Nz = input[0][0].size();

    std::vector<T> result(Nx * Ny * Nz);
    for (size_t i = 0; i < Nx; i++) {
        assert(input[i].size() == Ny);
        for (size_t j = 0; j < Ny; j++) {
            assert(input[i][j].size() == Nz);
            for (size_t k = 0; k < Nz; k++) {
                result[i + j * Nx + k * Nx * Ny] = input[i][j][k];
            }
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

template <typename T>
std::vector<std::vector<std::vector<T>>> ripple(const std::vector<T>& input, size_t Nx, size_t Ny) {
    assert(input.size() % Nx == 0);
    size_t Nz = input.size() / Nx / Ny;

    std::vector<std::vector<std::vector<T>>> result(Nx);

    for (size_t i = 0; i < Nx; i++) {
        result[i].resize(Ny);
        for (size_t j = 0; j < Ny; j++) {
            result[i][j].resize(Nz);
            for (size_t k = 0; k < Nz; k++) {
                // result[i + j * Nx + k * Nx * Ny] = input[i][j][k];
                result[i][j][k] = input[i + j * Nx + k * Nx * Ny];
            }
        }
    }
    return result;
}

// TODO: this is only for TV=double2 and T=double, 三维程序需要将此函数特化。
template <typename T, typename TV>
std::tuple<std::vector<T>, std::vector<T>> split(std::vector<TV> a) {
    CHECK_F(false, "Not implemented for this template type.");
}

inline std::tuple<std::vector<double>, std::vector<double>> split(std::vector<double2> a) {
    std::vector<double> a1(a.size());
    std::vector<double> a2(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        a1[i] = a[i].x;
        a2[i] = a[i].y;
    }
    return std::make_tuple(a1, a2);
}

inline std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> split(std::vector<double3> a) {
    std::vector<double> a1(a.size());
    std::vector<double> a2(a.size());
    std::vector<double> a3(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        a1[i] = a[i].x;
        a2[i] = a[i].y;
        a3[i] = a[i].z;
    }
    return std::make_tuple(a1, a2, a3);
}

// TODO: this is only for TV=double2 and T=double
template <typename T = double, typename TV = double2>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<T>>> split(std::vector<std::vector<TV>> a) {
    std::vector<std::vector<T>> a1(a.size());
    std::vector<std::vector<T>> a2(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        a1[i].resize(a[i].size());
        a2[i].resize(a[i].size());
        for (size_t j = 0; j < a[i].size(); j++) {
            a1[i][j] = a[i][j].x;
            a2[i][j] = a[i][j].y;
        }
    }
    return std::make_tuple(a1, a2);
}

// TODO: this is only for TV=double2 and T=double
template <typename T = double, typename TV = double2>
std::vector<TV> merge(std::vector<T> a1, std::vector<T> a2) {
    CHECK_F(a1.size() == a2.size());
    std::vector<double2> a(a1.size());
    for (size_t i = 0; i < a.size(); i++) {
        a[i].x = a1[i];
        a[i].y = a2[i];
    }
    return a;
}

template <typename T = double, typename TV = double3>
std::vector<TV> merge(std::vector<T> a1, std::vector<T> a2, std::vector<T> a3) {
    CHECK_F(a1.size() == a2.size());
    CHECK_F(a1.size() == a3.size());
    std::vector<double3> a(a1.size());
    for (size_t i = 0; i < a.size(); i++) {
        a[i].x = a1[i];
        a[i].y = a2[i];
        a[i].z = a3[i];
    }
    return a;
}

} // namespace algebra

#endif // __ALGEBRA_H__