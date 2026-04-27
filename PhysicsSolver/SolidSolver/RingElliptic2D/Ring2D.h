/**
 * @file Ring2D.h
 * @author Ma Pengfei (code@pengfeima.cn)
 * @brief
 * @version 0.1
 * @date 2023-11-09
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLaw.h"
#include "EnergyNorm.h"

namespace dolfin {
// 牛顿迭代法函数模板
template <typename Func, typename Derivative>
double newtonMethod(Func f, Derivative df, double initialGuess, double epsilon = 1e-8, int maxIterations = 100) {
    double x = initialGuess;
    for (int i = 0; i < maxIterations; ++i) {
        double fx  = f(x);
        double dfx = df(x);
        // cout << "x = " << x << ", f(x) = " << fx << ", f'(x) = " << dfx << endl;

        // 避免除以零
        if (abs(dfx) < 1e-6) {
            // cerr << "Error: Division by zero." << endl;
            return NAN; // 返回NaN表示无效的结果
        }

        // 牛顿迭代公式
        x = x - fx / dfx;
        // cout << "Iteration " << i + 1 << " : " << x << endl;

        // 判断是否足够接近零点
        if (abs(fx) < epsilon) {
            // cout << "Converged after " << i + 1 << " iterations." << endl;
            return x;
        }
    }

    // 牛顿迭代公式
    x = x - fx / dfx;
    // cout << "Iteration " << i + 1 << " : " << x << endl;

    // 判断是否足够接近零点
    if (abs(fx) < epsilon) {
        // cout << "Converged after " << i + 1 << " iterations." << endl;
        return x;
    }
}
return NAN; // 返回NaN表示无效的结果
}

void fun(const double& X0, const double& X1, double& s1, double& s2) {
    double gamma = 0.15;
    double R     = 0.25;
    auto   f     = [X0, X1, R, gamma](double s2) {
        double a = (X0 - 0.5);
        double b = (X1 - 0.5);
        double c = R + s2;
        double d = R + s2 + gamma;
        return 1 - (a / c) * (a / c) - (b / d) * (b / d);
    };

    auto df = [X0, X1, R, gamma](double s2) {
        double a = X0 - 0.5;
        double b = X1 - 0.5;
        double c = R + s2;
        double d = R + s2 + gamma;
        return 2 * (a * a / c / c / c + b * b / d / d / d);
    };

    // 初始猜测值[0.0.0625]
    double initialGuess = -0.03;

    // 使用函数模板进行牛顿迭代
    s2 = newtonMethod(f, df, initialGuess);

    if (!isnan(s2)) {
        // cout << "Approximate root: " << s2 << endl;
    }

    s1 = std::acos((X0 - 0.5) / (R + s2)) * R;
    isnan(s1) ? s1 = 1.0 : s1 = s1; // arccos(?) = nan when ? > 1
    X1 < 0.5 ? s1 = 2.0 * M_PI* R - s1 : s1 = s1;
}
class ParamConfiguration : public Expression {
  public:
    ParamConfiguration() : Expression(2) {}

    void eval(Array<double>& s, const Array<double>& x) const {
        fun(x[0], x[1], s[0], s[1]);
        printf("%f %f %f %f\n", x[0], x[1], s[0], s[1]);
    }
};

class Ring2D : public GenericSolidSolver<ConstitutiveLaw::FunctionSpace, ConstitutiveLaw::BilinearForm,
                                         ConstitutiveLaw::LinearForm> {
  public:
    std::shared_ptr<Expression> param_configuration;
    Ring2D(std::shared_ptr<Mesh> mesh, std::string result_path = "") : GenericSolidSolver(mesh, result_path) {
        param_configuration = std::make_shared<ParamConfiguration>();
        LOG_F(INFO, "2D Ring2D is called!");
    }

    double energy_norm(const std::vector<double>& vector_X) {
        auto X = std::make_shared<Function>(V);
        X->vector()->set_local(vector_X);
        EnergyNorm::Form_M3 energy_norm(_mesh);
        energy_norm.X = X;
        LOG_F(INFO, "Assembling energy norm...");
        double result = assemble(energy_norm);
        LOG_F(INFO, "弹性势能 : %.16e .", result);
        return result;
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->s = param_configuration;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }
};
} // namespace dolfin