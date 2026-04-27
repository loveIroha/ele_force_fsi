
#include <cmath>
#include <iostream>
#include <vector>

std::vector<double> restrict2D(std::vector<double> u) {
    auto num = static_cast<int>(sqrt(u.size()));

    std::vector<double> v((num / 2) * (num / 2));
    for (size_t i = 0; i < num / 2; i++) {
        for (size_t j = 0; j < num / 2; j++) {
            v[i * (num / 2) + j]
                = (u[2 * i * num + 2 * j + 1] + u[(2 * i + 1) * num + 2 * j] + u[(2 * i + 1) * num + 2 * j + 1] * 4.0
                   + u[(2 * i + 1) * num + 2 * j + 2] + u[(2 * i + 2) * num + 2 * j + 1])
                  / 8.0;
        }
    }
    return v;
}

std::vector<double> interpolate2D(std::vector<double> u) {
    auto num = static_cast<int>(sqrt(u.size()));

    std::vector<double> v((num * 2 + 1) * (num * 2 + 1));

    /// originals
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) {
            v[(2 * i + 1) * (num * 2 + 1) + (2 * j + 1)] = u[i * num + j];
        }
    }

    /// centers
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) {
            v[(2 * i + 2) * (num * 2 + 1) + (2 * j + 1)] = 0.5 * (u[i * num + j] + u[(i + 1) * num + j]);
            v[(2 * i + 1) * (num * 2 + 1) + (2 * j + 2)] = 0.5 * (u[i * num + j] + u[i * num + j + 1]);
        }
    }

    /// neighbours
    for (size_t i = 0; i < num - 1; i++) {
        for (size_t j = 0; j < num - 1; j++) {
            v[(2 * i + 2) * (num * 2 + 1) + (2 * j + 2)]
                = 0.25 * (u[i * num + j] + u[i * num + j + 1] + u[(i + 1) * num + j] + u[(i + 1) * num + j + 1]);
        }
    }

    /// edges
    for (size_t i = 1; i < 2 * num; i++) {
        v[i]                             = 0.5 * v[2 * num + 1 + i];
        v[(2 * num + 1) * i]             = 0.5 * v[(2 * num + 1) * i + 1];
        v[(2 * num + 1) * i + 2 * num]   = 0.5 * v[(2 * num + 1) * i + 2 * num - 1];
        v[(2 * num + 1) * (2 * num) + i] = 0.5 * v[(2 * num + 1) * (2 * num - 1) + i];
    }

    /// corners
    v[0]                                 = 0.25 * v[2 * num + 2];
    v[2 * num]                           = 0.25 * v[2 * num + 2 * num];
    v[(2 * num + 1) * 2 * num]           = 0.25 * v[(2 * num - 1) * (2 * num + 1) + 1];
    v[(2 * num + 1) * (2 * num + 1) - 1] = 0.25 * v[(2 * num) * (2 * num + 1) - 2];
    return v;
}

void print_vector_2D(std::vector<double> u) {
    int num = static_cast<int>(sqrt(u.size()));
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) {
            std::cout << u[i * num + j] << " ";
        }
        std::cout << std::endl;
    }
}

/// int main(){
int test() {
    // create the initial vector.
    int                 num = 7;
    std::vector<double> u(num * num);
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) {
            u[i * num + j] = i * num + j + 1;
        }
    }
    auto v = interpolate2D(u);
    auto w = restrict2D(u);

    print_vector_2D(u);
    print_vector_2D(v);
    print_vector_2D(w);

    return 0;
}

class NonlinearMultigridSolver2D {
  private:
    double gammaa = 10.0;
    double pi     = 3.14159265358979;

  public:
    double exact_solution_0(const double* x) { return sin(pi * x[1]) * sin(pi * x[0]); }

    double exact_solution_1(const double* x) { return sin(pi * x[1]) * sin(pi * x[0]); }

    double source_0(const double* x) {
        return pi * sin(pi * x[0]) * sin(pi * x[1]) * (2 * pi + gammaa * sin(pi * (x[0] + x[1])));
    }

    double source_1(const double* x) {
        return pi * sin(pi * x[0]) * sin(pi * x[1]) * (2 * pi + gammaa * sin(pi * (x[0] + x[1])));
    }

    std::vector<std::vector<double>> exact_solution_vector(const std::vector<std::vector<double>> x) {
        auto mesh_size = x[0].size();
        auto num       = static_cast<int>(sqrt(mesh_size));

        /// remember old values.
        std::vector<std::vector<double>> u(2);
        u[0].resize(mesh_size);
        u[1].resize(mesh_size);

        for (size_t i = 0; i < num; i++) {
            for (size_t j = 0; j < num; j++) {
                double position[] = {x[0][i * num + j], x[1][i * num + j]};
                u[0][i * num + j] = exact_solution_0(position);
                u[1][i * num + j] = exact_solution_1(position);
            }
        }
        return u;
    }

    /// TODO:
    /// 一维问题可以直接将u当成变量。当u是二维向量时，u=(u,v)怎么避免变量重复？
    std::vector<std::vector<double>> residual_vector(std::vector<std::vector<double>>& u_symbol,
                                                     std::vector<std::vector<double>>& x) {
        auto& u = u_symbol[0];
        auto& v = u_symbol[1];

        auto mesh_size = u.size();
        auto num       = static_cast<int>(sqrt(mesh_size));
        auto h         = 1.0 / (num + 1);

        std::vector<std::vector<double>> r(2);
        r[0].resize(mesh_size);
        r[1].resize(mesh_size);

        /// local data
        double u_elements[5];
        double v_elements[5];
        double position[2];

        /// 遍历，计算残差
        for (size_t i = 1; i < num - 1; i++) {
            for (size_t j = 1; j < num - 1; j++) {
                u_elements[0] = u[i * num + j];
                u_elements[1] = u[(i - 1) * num + j];
                u_elements[2] = u[(i + 1) * num + j];
                u_elements[3] = u[i * num + j - 1];
                u_elements[4] = u[i * num + j + 1];
                v_elements[0] = v[i * num + j];
                v_elements[1] = v[(i - 1) * num + j];
                v_elements[2] = v[(i + 1) * num + j];
                v_elements[3] = v[i * num + j - 1];
                v_elements[4] = v[i * num + j + 1];

                /// residual_equation_0(u,v,)
                /// r[i] = residual(&(u[i-1]), source(x[i]), h);
                position[0] = x[0][i * num + j];
                position[1] = x[1][i * num + j];

                r[0][i * num + j] = residual_0(u_elements, v_elements, source_0(position), h);
                r[1][i * num + j] = residual_1(u_elements, v_elements, source_1(position), h);
            }
        }
        /// update four edges
        for (size_t i = 1; i < num - 1; i++) {
            position[0] = x[0][0 * num + i];
            position[1] = x[1][0 * num + i];

            u_elements[0] = u[0 * num + i];
            u_elements[1] = 0.0;
            u_elements[2] = u[1 * num + i];
            u_elements[3] = u[i - 1];
            u_elements[4] = u[i + 1];
            v_elements[0] = v[0 * num + i];
            v_elements[1] = 0.0;
            v_elements[2] = v[1 * num + i];
            v_elements[3] = v[i - 1];
            v_elements[4] = v[i + 1];

            r[0][0 * num + i] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][0 * num + i] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num * i + 0];
            position[1] = x[1][num * i + 0];

            u_elements[0] = u[num * i + 0];
            u_elements[1] = u[num * (i - 1) + 0];
            u_elements[2] = u[num * (i + 1) + 0];
            u_elements[3] = 0.0;
            u_elements[4] = u[num * i + 1];
            v_elements[0] = v[num * i + 0];
            v_elements[1] = v[num * (i - 1) + 0];
            v_elements[2] = v[num * (i + 1) + 0];
            v_elements[3] = 0.0;
            v_elements[4] = v[num * i + 1];

            r[0][num * i + 0] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num * i + 0] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num * i + num - 1];
            position[1] = x[1][num * i + num - 1];

            u_elements[0] = u[num * i + num - 1];
            u_elements[1] = u[num * (i - 1) + num - 1];
            u_elements[2] = u[num * (i + 1) + num - 1];
            u_elements[3] = u[num * i + num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * i + num - 1];
            v_elements[1] = v[num * (i - 1) + num - 1];
            v_elements[2] = v[num * (i + 1) + num - 1];
            v_elements[3] = v[num * i + num - 2];
            v_elements[4] = 0.0;

            r[0][num * i + num - 1] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num * i + num - 1] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num * (num - 1) + i];
            position[1] = x[1][num * (num - 1) + i];

            u_elements[0] = u[num * (num - 1) + i];
            u_elements[1] = u[num * (num - 2) + i];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * (num - 1) + i - 1];
            u_elements[4] = u[num * (num - 1) + i + 1];
            v_elements[0] = v[num * (num - 1) + i];
            v_elements[1] = v[num * (num - 2) + i];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * (num - 1) + i - 1];
            v_elements[4] = v[num * (num - 1) + i + 1];

            r[0][num * (num - 1) + i] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num * (num - 1) + i] = residual_1(u_elements, v_elements, source_1(position), h);
        }

        {
            position[0] = x[0][0];
            position[1] = x[1][0];

            u_elements[0] = u[0];
            u_elements[1] = 0.0;
            u_elements[2] = u[num];
            u_elements[3] = 0.0;
            u_elements[4] = u[1];
            v_elements[0] = v[0];
            v_elements[1] = 0.0;
            v_elements[2] = v[num];
            v_elements[3] = 0.0;
            v_elements[4] = u[1];

            r[0][0] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][0] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num - 1];
            position[1] = x[1][num - 1];

            u_elements[0] = u[num - 1];
            u_elements[1] = 0.0;
            u_elements[2] = u[2 * num - 1];
            u_elements[3] = u[num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num - 1];
            v_elements[1] = 0.0;
            v_elements[2] = v[2 * num - 1];
            v_elements[3] = v[num - 2];
            v_elements[4] = 0.0;

            r[0][num - 1] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num - 1] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num * num - 1];
            position[1] = x[1][num * num - 1];

            u_elements[0] = u[num * num - 1];
            u_elements[1] = u[num * (num - 1) - 1];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * num - 1];
            v_elements[1] = v[num * (num - 1) - 1];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * num - 2];
            v_elements[4] = 0.0;

            r[0][num * num - 1] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num * num - 1] = residual_1(u_elements, v_elements, source_1(position), h);

            position[0] = x[0][num * num - num];
            position[1] = x[1][num * num - num];

            u_elements[0] = u[num * num - num];
            u_elements[1] = u[num * (num - 1) - num];
            u_elements[2] = 0.0;
            u_elements[3] = 0.0;
            u_elements[4] = u[num * num - num + 1];
            v_elements[0] = v[num * num - num];
            v_elements[1] = v[num * (num - 1) - num];
            v_elements[2] = 0.0;
            v_elements[3] = 0.0;
            v_elements[4] = v[num * num - num + 1];

            r[0][num * num - num] = residual_0(u_elements, v_elements, source_0(position), h);
            r[1][num * num - num] = residual_1(u_elements, v_elements, source_1(position), h);
        }

        return r;
    }

    /// 标量的拉普拉斯算子
    /// 先约定元素的排列方式.总共五个元素，按照空间的排列，顺序如下：
    /// [u_{i, j}, u_{i-1, j}, u_{i+1, j}, u_{i, j-1}, u_{i, j+1}]

    double laplace_scalar(const double* u, double h) { return (u[1] + u[2] + u[3] + u[4] - 4.0 * u[0]) / h / h; }

    /// 二维对流项
    /// 数组u和数组v的排列方式和前面一样。
    double convection_0(const double* u, const double* v, double h) {
        return u[0] * (u[2] - u[1]) / 2.0 / h + v[0] * (u[4] - u[3]) / 2.0 / h;
    }

    double convection_1(const double* u, const double* v, double h) {
        return u[0] * (v[2] - v[1]) / 2.0 / h + v[0] * (v[4] - v[3]) / 2.0 / h;
    }

    double nonlinear_operator_0(const double* u, const double* v, double h) {
        return -laplace_scalar(u, h) + gammaa * convection_0(u, v, h);
    }

    double nonlinear_operator_1(const double* u, const double* v, double h) {
        return -laplace_scalar(v, h) + gammaa * convection_1(u, v, h);
    }

    double residual_0(const double* u, const double* v, double f, double h) {
        return nonlinear_operator_0(u, v, h) - f;
    }

    double residual_1(const double* u, const double* v, double f, double h) {
        return nonlinear_operator_1(u, v, h) - f;
    }

    double residual_equation_0(const double* u, const double* v, const double* u_old, const double* v_old, double r,
                               double h) {
        return nonlinear_operator_0(u, v, h) - nonlinear_operator_0(u_old, v_old, h) + r;
    }

    double residual_equation_1(const double* u, const double* v, const double* u_old, const double* v_old, double r,
                               double h) {
        return nonlinear_operator_1(u, v, h) - nonlinear_operator_1(u_old, v_old, h) + r;
    }

    // 牛顿迭代法求解非线性标量方程
    double newton_scalar_solver_0(const double* u, const double* v, const double* u_old, const double* v_old, double r,
                                  double h, double tol = 1e-7, int max_iter = 100, double delta = 1e-7) {
        double u_now = u[0];
        double u_next;

        for (size_t i = 0; i < max_iter; i++) {
            double array_u[]       = {u_now, u[1], u[2], u[3], u[4]};
            double array_u_delta[] = {u_now + delta, u[1], u[2], u[3], u[4]};

            double h_u       = residual_equation_0(array_u, v, u_old, v_old, r, h);
            double h_u_delta = residual_equation_0(array_u_delta, v, u_old, v_old, r, h);

            u_next = u_now - h_u * delta / (h_u_delta - h_u);
            if (fabs(u_now - u_next) <= tol) break;
            u_now = u_next;
        }
        return u_next;
    }

    // 牛顿迭代法求解非线性标量方程
    double newton_scalar_solver_1(const double* u, const double* v, const double* u_old, const double* v_old, double r,
                                  double h, double tol = 1e-7, int max_iter = 100, double delta = 1e-7) {
        double v_now = v[0];
        double v_next;
        for (size_t i = 0; i < max_iter; i++) {
            double array_v[]       = {v_now, v[1], v[2], v[3], v[4]};
            double array_v_delta[] = {v_now + delta, v[1], v[2], v[3], v[4]};

            double h_v       = residual_equation_1(u, array_v, u_old, v_old, r, h);
            double h_v_delta = residual_equation_1(u, array_v_delta, u_old, v_old, r, h);

            v_next = v_now - h_v * delta / (h_v_delta - h_v);

            if (fabs(v_now - v_next) <= tol) break;
            v_now = v_next;
        }
        return v_next;
    }

    // 通过差分法，用精确解计算出右端项。
    std::vector<std::vector<double>> source_approximation(std::vector<std::vector<double>>& u_exact) {
        auto& u         = u_exact[0];
        auto& v         = u_exact[1];
        auto  mesh_size = u.size();

        std::vector<std::vector<double>> f(2);
        f[0].resize(mesh_size);
        f[1].resize(mesh_size);

        auto   num = static_cast<int>(sqrt(u.size()));
        double h   = 1.0 / (num + 1);

        double u_elements[5];
        double v_elements[5];

        for (size_t i = 1; i < num - 1; i++) {
            for (size_t j = 1; j < num - 1; j++) {
                u_elements[0] = u[i * num + j];
                u_elements[1] = u[(i - 1) * num + j];
                u_elements[2] = u[(i + 1) * num + j];
                u_elements[3] = u[i * num + j - 1];
                u_elements[4] = u[i * num + j + 1];
                v_elements[0] = v[i * num + j];
                v_elements[1] = v[(i - 1) * num + j];
                v_elements[2] = v[(i + 1) * num + j];
                v_elements[3] = v[i * num + j - 1];
                v_elements[4] = v[i * num + j + 1];

                f[0][i * num + j] = nonlinear_operator_0(u_elements, v_elements, h);
                f[1][i * num + j] = nonlinear_operator_1(u_elements, v_elements, h);
            }
        }

        /// update four edges
        for (size_t i = 1; i < num - 1; i++) {
            u_elements[0] = u[0 * num + i];
            u_elements[1] = 0.0;
            u_elements[2] = u[1 * num + i];
            u_elements[3] = u[i - 1];
            u_elements[4] = u[i + 1];
            v_elements[0] = v[0 * num + i];
            v_elements[1] = 0.0;
            v_elements[2] = v[1 * num + i];
            v_elements[3] = v[i - 1];
            v_elements[4] = v[i + 1];

            f[0][0 * num + i] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][0 * num + i] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num * i + 0];
            u_elements[1] = u[num * (i - 1) + 0];
            u_elements[2] = u[num * (i + 1) + 0];
            u_elements[3] = 0.0;
            u_elements[4] = u[num * i + 1];
            v_elements[0] = v[num * i + 0];
            v_elements[1] = v[num * (i - 1) + 0];
            v_elements[2] = v[num * (i + 1) + 0];
            v_elements[3] = 0.0;
            v_elements[4] = v[num * i + 1];

            f[0][num * i + 0] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num * i + 0] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num * i + num - 1];
            u_elements[1] = u[num * (i - 1) + num - 1];
            u_elements[2] = u[num * (i + 1) + num - 1];
            u_elements[3] = u[num * i + num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * i + num - 1];
            v_elements[1] = v[num * (i - 1) + num - 1];
            v_elements[2] = v[num * (i + 1) + num - 1];
            v_elements[3] = v[num * i + num - 2];
            v_elements[4] = 0.0;

            f[0][num * i + num - 1] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num * i + num - 1] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num * (num - 1) + i];
            u_elements[1] = u[num * (num - 2) + i];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * (num - 1) + i - 1];
            u_elements[4] = u[num * (num - 1) + i + 1];
            v_elements[0] = v[num * (num - 1) + i];
            v_elements[1] = v[num * (num - 2) + i];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * (num - 1) + i - 1];
            v_elements[4] = v[num * (num - 1) + i + 1];

            f[0][num * (num - 1) + i] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num * (num - 1) + i] = nonlinear_operator_1(u_elements, v_elements, h);
        }

        /// update four corners
        {
            u_elements[0] = u[0];
            u_elements[1] = 0.0;
            u_elements[2] = u[num];
            u_elements[3] = 0.0;
            u_elements[4] = u[1];
            v_elements[0] = v[0];
            v_elements[1] = 0.0;
            v_elements[2] = v[num];
            v_elements[3] = 0.0;
            v_elements[4] = u[1];

            f[0][0] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][0] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num - 1];
            u_elements[1] = 0.0;
            u_elements[2] = u[2 * num - 1];
            u_elements[3] = u[num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num - 1];
            v_elements[1] = 0.0;
            v_elements[2] = v[2 * num - 1];
            v_elements[3] = v[num - 2];
            v_elements[4] = 0.0;

            f[0][num - 1] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num - 1] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num * num - 1];
            u_elements[1] = u[num * (num - 1) - 1];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * num - 1];
            v_elements[1] = v[num * (num - 1) - 1];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * num - 2];
            v_elements[4] = 0.0;

            f[0][num * num - 1] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num * num - 1] = nonlinear_operator_1(u_elements, v_elements, h);

            u_elements[0] = u[num * num - num];
            u_elements[1] = u[num * (num - 1) - num];
            u_elements[2] = 0.0;
            u_elements[3] = 0.0;
            u_elements[4] = u[num * num - num + 1];
            v_elements[0] = v[num * num - num];
            v_elements[1] = v[num * (num - 1) - num];
            v_elements[2] = 0.0;
            v_elements[3] = 0.0;
            v_elements[4] = v[num * num - num + 1];

            f[0][num * num - num] = nonlinear_operator_0(u_elements, v_elements, h);
            f[1][num * num - num] = nonlinear_operator_1(u_elements, v_elements, h);
        }
        return f;
    }

    /// std::vector<std::vector<double>> r
    void nonlinear_gauss_seidel_iterator(std::vector<std::vector<double>>& result, std::vector<std::vector<double>> r) {
        auto& u         = result[0];
        auto& v         = result[1];
        auto  mesh_size = u.size();

        std::vector<double> u_old(mesh_size);
        std::vector<double> v_old(mesh_size);
        for (size_t i = 0; i < mesh_size; i++) {
            v_old[i] = v[i];
            u_old[i] = u[i];
        }

        auto   num = static_cast<int>(sqrt(mesh_size));
        double h   = 1.0 / (num + 1);

        double u_elements[5];
        double v_elements[5];
        double u_old_elements[5];
        double v_old_elements[5];

        for (size_t i = 1; i < num - 1; i++) {
            for (size_t j = 1; j < num - 1; j++) {
                u_elements[0] = u[i * num + j];
                u_elements[1] = u[(i - 1) * num + j];
                u_elements[2] = u[(i + 1) * num + j];
                u_elements[3] = u[i * num + j - 1];
                u_elements[4] = u[i * num + j + 1];
                v_elements[0] = v[i * num + j];
                v_elements[1] = v[(i - 1) * num + j];
                v_elements[2] = v[(i + 1) * num + j];
                v_elements[3] = v[i * num + j - 1];
                v_elements[4] = v[i * num + j + 1];

                u_old_elements[0] = u_old[i * num + j];
                u_old_elements[1] = u_old[(i - 1) * num + j];
                u_old_elements[2] = u_old[(i + 1) * num + j];
                u_old_elements[3] = u_old[i * num + j - 1];
                u_old_elements[4] = u_old[i * num + j + 1];
                v_old_elements[0] = v_old[i * num + j];
                v_old_elements[1] = v_old[(i - 1) * num + j];
                v_old_elements[2] = v_old[(i + 1) * num + j];
                v_old_elements[3] = v_old[i * num + j - 1];
                v_old_elements[4] = v_old[i * num + j + 1];

                u[i * num + j] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements,
                                                        r[0][i * num + j], h);
                v[i * num + j] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements,
                                                        r[1][i * num + j], h);
            }
        }

        /// update four edges
        for (size_t i = 1; i < num - 1; i++) {
            u_elements[0] = u[0 * num + i];
            u_elements[1] = 0.0;
            u_elements[2] = u[1 * num + i];
            u_elements[3] = u[i - 1];
            u_elements[4] = u[i + 1];
            v_elements[0] = v[0 * num + i];
            v_elements[1] = 0.0;
            v_elements[2] = v[1 * num + i];
            v_elements[3] = v[i - 1];
            v_elements[4] = v[i + 1];

            u_old_elements[0] = u_old[0 * num + i];
            u_old_elements[1] = 0.0;
            u_old_elements[2] = u_old[1 * num + i];
            u_old_elements[3] = u_old[i - 1];
            u_old_elements[4] = u_old[i + 1];
            v_old_elements[0] = v_old[0 * num + i];
            v_old_elements[1] = 0.0;
            v_old_elements[2] = v_old[1 * num + i];
            v_old_elements[3] = v_old[i - 1];
            v_old_elements[4] = v_old[i + 1];

            u[0 * num + i]
                = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements, r[0][0 * num + i], h);
            v[0 * num + i]
                = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements, r[1][0 * num + i], h);

            u_elements[0] = u[num * i + 0];
            u_elements[1] = u[num * (i - 1) + 0];
            u_elements[2] = u[num * (i + 1) + 0];
            u_elements[3] = 0.0;
            u_elements[4] = u[num * i + 1];
            v_elements[0] = v[num * i + 0];
            v_elements[1] = v[num * (i - 1) + 0];
            v_elements[2] = v[num * (i + 1) + 0];
            v_elements[3] = 0.0;
            v_elements[4] = v[num * i + 1];

            u_old_elements[0] = u_old[num * i + 0];
            u_old_elements[1] = u_old[num * (i - 1) + 0];
            u_old_elements[2] = u_old[num * (i + 1) + 0];
            u_old_elements[3] = 0.0;
            u_old_elements[4] = u_old[num * i + 1];
            v_old_elements[0] = v_old[num * i + 0];
            v_old_elements[1] = v_old[num * (i - 1) + 0];
            v_old_elements[2] = v_old[num * (i + 1) + 0];
            v_old_elements[3] = 0.0;
            v_old_elements[4] = v_old[num * i + 1];

            u[i * num + 0]
                = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements, r[0][i * num + 0], h);
            v[i * num + 0]
                = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements, r[1][i * num + 0], h);

            u_elements[0] = u[num * i + num - 1];
            u_elements[1] = u[num * (i - 1) + num - 1];
            u_elements[2] = u[num * (i + 1) + num - 1];
            u_elements[3] = u[num * i + num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * i + num - 1];
            v_elements[1] = v[num * (i - 1) + num - 1];
            v_elements[2] = v[num * (i + 1) + num - 1];
            v_elements[3] = v[num * i + num - 2];
            v_elements[4] = 0.0;

            u_old_elements[0] = u_old[num * i + num - 1];
            u_old_elements[1] = u_old[num * (i - 1) + num - 1];
            u_old_elements[2] = u_old[num * (i + 1) + num - 1];
            u_old_elements[3] = u_old[num * i + num - 2];
            u_old_elements[4] = 0.0;
            v_old_elements[0] = v_old[num * i + num - 1];
            v_old_elements[1] = v_old[num * (i - 1) + num - 1];
            v_old_elements[2] = v_old[num * (i + 1) + num - 1];
            v_old_elements[3] = v_old[num * i + num - 2];
            v_old_elements[4] = 0.0;

            u[num * i + num - 1] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements,
                                                          r[0][num * i + num - 1], h);
            v[num * i + num - 1] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements,
                                                          r[1][num * i + num - 1], h);

            u_elements[0] = u[num * (num - 1) + i];
            u_elements[1] = u[num * (num - 2) + i];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * (num - 1) + i - 1];
            u_elements[4] = u[num * (num - 1) + i + 1];
            v_elements[0] = v[num * (num - 1) + i];
            v_elements[1] = v[num * (num - 2) + i];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * (num - 1) + i - 1];
            v_elements[4] = v[num * (num - 1) + i + 1];

            u_old_elements[0] = u_old[num * (num - 1) + i];
            u_old_elements[1] = u_old[num * (num - 2) + i];
            u_old_elements[2] = 0.0;
            u_old_elements[3] = u_old[num * (num - 1) + i - 1];
            u_old_elements[4] = u_old[num * (num - 1) + i + 1];
            v_old_elements[0] = v_old[num * (num - 1) + i];
            v_old_elements[1] = v_old[num * (num - 2) + i];
            v_old_elements[2] = 0.0;
            v_old_elements[3] = v_old[num * (num - 1) + i - 1];
            v_old_elements[4] = v_old[num * (num - 1) + i + 1];

            u[num * (num - 1) + i] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements,
                                                            r[0][num * (num - 1) + i], h);
            v[num * (num - 1) + i] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements,
                                                            r[1][num * (num - 1) + i], h);
        }

        {
            u_elements[0] = u[0];
            u_elements[1] = 0.0;
            u_elements[2] = u[num];
            u_elements[3] = 0.0;
            u_elements[4] = u[1];
            v_elements[0] = v[0];
            v_elements[1] = 0.0;
            v_elements[2] = v[num];
            v_elements[3] = 0.0;
            v_elements[4] = u[1];

            u_old_elements[0] = u_old[0];
            u_old_elements[1] = 0.0;
            u_old_elements[2] = u_old[num];
            u_old_elements[3] = 0.0;
            u_old_elements[4] = u_old[1];
            v_old_elements[0] = v_old[0];
            v_old_elements[1] = 0.0;
            v_old_elements[2] = v_old[num];
            v_old_elements[3] = 0.0;
            v_old_elements[4] = u_old[1];

            u[0] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements, r[0][0], h);
            v[0] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements, r[1][0], h);

            u_elements[0] = u[num - 1];
            u_elements[1] = 0.0;
            u_elements[2] = u[2 * num - 1];
            u_elements[3] = u[num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num - 1];
            v_elements[1] = 0.0;
            v_elements[2] = v[2 * num - 1];
            v_elements[3] = v[num - 2];
            v_elements[4] = 0.0;

            u_old_elements[0] = u_old[num - 1];
            u_old_elements[1] = 0.0;
            u_old_elements[2] = u_old[2 * num - 1];
            u_old_elements[3] = u_old[num - 2];
            u_old_elements[4] = 0.0;
            v_old_elements[0] = v_old[num - 1];
            v_old_elements[1] = 0.0;
            v_old_elements[2] = v_old[2 * num - 1];
            v_old_elements[3] = v_old[num - 2];
            v_old_elements[4] = 0.0;

            u[num - 1]
                = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements, r[0][num - 1], h);
            v[num - 1]
                = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements, r[1][num - 1], h);

            u_elements[0] = u[num * num - 1];
            u_elements[1] = u[num * (num - 1) - 1];
            u_elements[2] = 0.0;
            u_elements[3] = u[num * num - 2];
            u_elements[4] = 0.0;
            v_elements[0] = v[num * num - 1];
            v_elements[1] = v[num * (num - 1) - 1];
            v_elements[2] = 0.0;
            v_elements[3] = v[num * num - 2];
            v_elements[4] = 0.0;

            u_old_elements[0] = u_old[num * num - 1];
            u_old_elements[1] = u_old[num * (num - 1) - 1];
            u_old_elements[2] = 0.0;
            u_old_elements[3] = u_old[num * num - 2];
            u_old_elements[4] = 0.0;
            v_old_elements[0] = v_old[num * num - 1];
            v_old_elements[1] = v_old[num * (num - 1) - 1];
            v_old_elements[2] = 0.0;
            v_old_elements[3] = v_old[num * num - 2];
            v_old_elements[4] = 0.0;

            u[num * num - 1] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements,
                                                      r[0][num * num - 1], h);
            v[num * num - 1] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements,
                                                      r[1][num * num - 1], h);

            u_elements[0] = u[num * num - num];
            u_elements[1] = u[num * (num - 1) - num];
            u_elements[2] = 0.0;
            u_elements[3] = 0.0;
            u_elements[4] = u[num * num - num + 1];
            v_elements[0] = v[num * num - num];
            v_elements[1] = v[num * (num - 1) - num];
            v_elements[2] = 0.0;
            v_elements[3] = 0.0;
            v_elements[4] = v[num * num - num + 1];

            u_old_elements[0] = u_old[num * num - num];
            u_old_elements[1] = u_old[num * (num - 1) - num];
            u_old_elements[2] = 0.0;
            u_old_elements[3] = 0.0;
            u_old_elements[4] = u_old[num * num - num + 1];
            v_old_elements[0] = v_old[num * num - num];
            v_old_elements[1] = v_old[num * (num - 1) - num];
            v_old_elements[2] = 0.0;
            v_old_elements[3] = 0.0;
            v_old_elements[4] = v_old[num * num - num + 1];

            u[num * num - num] = newton_scalar_solver_0(u_elements, v_elements, u_old_elements, v_old_elements,
                                                        r[0][num * num - num], h);
            v[num * num - num] = newton_scalar_solver_1(u_elements, v_elements, u_old_elements, v_old_elements,
                                                        r[1][num * num - num], h);
        }
    }
    NonlinearMultigridSolver2D(/* args */);
    ~NonlinearMultigridSolver2D();
};

NonlinearMultigridSolver2D::NonlinearMultigridSolver2D(/* args */) {}

NonlinearMultigridSolver2D::~NonlinearMultigridSolver2D() {}

void two_grid_multigrid_2D(NonlinearMultigridSolver2D nms, std::vector<std::vector<double>>& u,
                           std::vector<std::vector<double>>& x) {
    auto mesh_size   = u[0].size();
    auto num         = static_cast<int>(sqrt(mesh_size));
    auto num_2       = num / 2;
    auto mesh_size_2 = num_2 * num_2;

    auto r = nms.residual_vector(u, x);
    nms.nonlinear_gauss_seidel_iterator(u, r);

    std::vector<std::vector<double>> r_2(2);
    std::vector<std::vector<double>> u_2(2);
    std::vector<std::vector<double>> v_2(2);
    r_2[0] = restrict2D(r[0]);
    r_2[1] = restrict2D(r[1]);
    u_2[0] = restrict2D(u[0]);
    u_2[1] = restrict2D(u[1]);
    v_2[0] = restrict2D(u[0]);
    v_2[1] = restrict2D(u[1]);

    double                           h_2 = 1.0 / (num_2 + 1);
    std::vector<std::vector<double>> x_2(2);
    x_2[0].resize(mesh_size_2);
    x_2[1].resize(mesh_size_2);

    for (size_t i = 0; i < num_2; i++) {
        for (size_t j = 0; j < num_2; j++) {
            x_2[0][i * num_2 + j] = h_2 * (i + 1);
            x_2[1][i * num_2 + j] = h_2 * (j + 1);
        }
    }

    if (num_2 <= 30) {
        /// 如果网格足够粗，那么进行高斯赛德尔迭代。
        for (size_t i = 0; i < 200; i++) {
            nms.nonlinear_gauss_seidel_iterator(u_2, r_2);
            r_2 = nms.residual_vector(u_2, x_2);
        }
    } else {
        /// 如果网格还是细，那么转换到下一层网格上计算。
        two_grid_multigrid_2D(nms, u_2, x_2);
    }
    /// 更深一层的网格

    /// e = interpolate(u_2 - v_2);
    std::vector<std::vector<double>> e_2(2);
    e_2[0].resize(mesh_size_2);
    e_2[1].resize(mesh_size_2);

    for (size_t i = 0; i < mesh_size_2; i++) {
        e_2[0][i] = u_2[0][i] - v_2[0][i];
        e_2[1][i] = u_2[1][i] - v_2[1][i];
    }
    std::vector<std::vector<double>> e(2);
    e[0] = interpolate2D(e_2[0]);
    e[1] = interpolate2D(e_2[1]);

    /// u = u + e
    for (size_t i = 0; i < mesh_size; i++) {
        u[0][i] = u[0][i] + e[0][i];
        u[1][i] = u[1][i] + e[1][i];
    }

    for (size_t i = 0; i < 1; i++) {
        r = nms.residual_vector(u, x);
        nms.nonlinear_gauss_seidel_iterator(u, r);
    }
}

double distance(std::vector<double> a, std::vector<double> b) {
    double sum = 0.0;
    ///  TODO: 判断a和b的长度是否相等
    for (size_t i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return sqrt(sum);
}

int main() {
    auto num       = 511;
    auto mesh_size = num * num;
    auto h         = 1.0 / (num + 1);

    std::vector<std::vector<double>> x(2);
    x[0].resize(mesh_size);
    x[1].resize(mesh_size);

    std::vector<std::vector<double>> v(2);
    v[0].resize(mesh_size);
    v[1].resize(mesh_size);

    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) {
            x[0][i * num + j] = h * (i + 1);
            x[1][i * num + j] = h * (j + 1);
        }
    }

    NonlinearMultigridSolver2D nms;

    // auto u = nms.exact_solution_vector(x);
    auto                             u_exact = nms.exact_solution_vector(x);
    std::vector<std::vector<double>> u(2);
    u[0].resize(mesh_size);
    u[1].resize(mesh_size);

    auto f         = nms.source_approximation(u);
    auto r         = nms.residual_vector(u, x);
    auto max_error = 0.0;
    for (size_t i = 0; i < 10; i++) {
        for (size_t i = 0; i < num; i++) {
            for (size_t j = 0; j < num; j++) {
                v[0][i * num + j] = u[0][i * num + j];
                v[1][i * num + j] = u[1][i * num + j];
            }
        }
        two_grid_multigrid_2D(nms, u, x);
    }
    return 0;
}