#include <cmath>
#include <functional>
#include <iostream>

using namespace std;

// 牛顿迭代法函数模板
template <typename Func, typename Derivative>
double newtonMethod(Func f, Derivative df, double initialGuess, double epsilon = 1e-8, int maxIterations = 100) {
    double x = initialGuess;

    for (int i = 0; i < maxIterations; ++i) {
        double fx  = f(x);
        double dfx = df(x);
        cout << "x = " << x << ", f(x) = " << fx << ", f'(x) = " << dfx << endl;

        // 避免除以零
        if (abs(dfx) < 1e-6) {
            cerr << "Error: Division by zero." << endl;
            return NAN; // 返回NaN表示无效的结果
        }

        // 牛顿迭代公式
        x = x - fx / dfx;
        cout << "Iteration " << i + 1 << " : " << x << endl;

        // 判断是否足够接近零点
        if (abs(fx) < epsilon) {
            cout << "Converged after " << i + 1 << " iterations." << endl;
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
    double initialGuess = 0.00;

    // 使用函数模板进行牛顿迭代
    s2 = newtonMethod(f, df, initialGuess);

    if (!isnan(s2)) { cout << "Approximate root: " << s2 << endl; }

    s1 = std::acos((X0 - 0.5) / (R + s2)) * R;
    X1 < 0.5 ? s1 = 2 * M_PI* R - s1 : s1 = s1;
}
int main() {
    // 定义要求根的函数和导数
    // 0.562050 0.741668
    double X0 = 0.208435;
    double X1 = 0.500000;
    double s1 = 0.0;
    double s2 = 0.0;
    fun(X0, X1, s1, s2);
    std::cout << s1 << " " << s2 << std::endl;

    return 0;
}
