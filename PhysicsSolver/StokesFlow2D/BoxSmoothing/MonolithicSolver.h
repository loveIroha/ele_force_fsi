

#pragma once
#include <Eigen/Dense>

class NavierStokesLocal2D {
    // private:
  public:
    double          hx;
    double          hy;
    double          dt;
    Eigen::MatrixXd matrix;
    Eigen::MatrixXd inverse;

  public:
    NavierStokesLocal2D(double hx, double hy, double dt, double mu, double rho) : hx(hx), hy(hy), dt(dt) {
        double a = 1.0 / hx / hx + 1.0 / hy / hy;
        double b = 1.0 / hx;
        double c = 1.0 / hy;
        double d = 1.0 / dt;
        matrix.resize(5, 5);

        // 设置矩阵的元素 方式一
        matrix.setConstant(0.0);
        Eigen::MatrixXd A11(2, 2);
        Eigen::MatrixXd A22(2, 2);
        Eigen::MatrixXd B(4, 1);

        A11 << rho * d + 2 * mu * a, -mu * b * b, -mu * b * b, rho * d + 2 * mu * a;
        A22 << rho * d + 2 * mu * a, -mu * c * c, -mu * c * c, rho * d + 2 * mu * a;
        B << b, -b, c, -c;

        Eigen::MatrixXd BT       = B.transpose();
        matrix.block(0, 0, 2, 2) = A11;
        matrix.block(2, 2, 2, 2) = A22;
        matrix.block(4, 0, 1, 4) = BT;
        matrix.block(0, 4, 4, 1) = B;

        inverse = matrix.inverse();
    };

    double operator()(int i, int j) {
        if (i >= 0 && i < matrix.rows() && j >= 0 && j < matrix.cols()) {
            return inverse(i, j);
        } else {
            // 可以添加适当的错误处理或者返回默认值
            // 这里简单返回0作为默认值
            return 0.0;
        }
    }

    void solve(double* x, const double* b, int n = 5) const {
        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            for (int j = 0; j < n; j++) {
                x[i] += inverse(j, i) * b[j];
            }
        }
    }

    ~NavierStokesLocal2D(){};
};
