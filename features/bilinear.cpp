

#include <array>
#include <functional>
#include <iostream>

// 考虑GPU设备运行，使用函数指针而不是lamgbda函数
double L0(double x) { return 1.0 - x; }
double L1(double x) { return x; }
double (*L[2])(double) = {L0, L1};

int test_linears() {
    double f[2];
    // f(x) = x+1
    f[0] = 0; // (0)
    f[1] = 1; // (1)

    // std::array<std::function<double(double)>, 2> L = {{
    //     [](double x) { return (1.0-x); },
    //     [](double x) { return (x); }
    // }};

    double sum = 0.0;
    double x   = 0.1;
    double y   = 0.3;
    for (int i = 0; i < 2; i++) {
        sum += f[i] * L[i](x);
        std::cout << sum << std::endl;
    }
    return 0;
}

template <int DIM = 2>
double bilinear(double x, double y, double* f) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            sum += f[j * DIM + i] * L[i](x) * L[j](y);
    return sum;
}

int main() {
    double f[4];
    // f(x,y) = x+2*y+2xy
    f[0] = 0; // (0,0)
    f[1] = 1; // (1,0)
    f[2] = 2; // (0,1)
    f[3] = 5; // (1,1)

    double x = 0.1;
    double y = 0.3;

    std::cout << x + 2 * y + 2 * x * y << std::endl;
    std::cout << bilinear<2>(x, y, f) << std::endl;

    return 0;
}

// 三线性插值
double f_(double x, double y, double z) { return x + y + z + x * y + x * z + y * z + x * y * z + 1; }
int    main_1() {
    double f[2][2][2];
    // f(x,y) = x+y+z+xy+xz+yz+xyz+1
    f[0][0][0] = f_(0, 0, 0); //
    f[1][0][0] = f_(1, 0, 0); //
    f[0][1][0] = f_(0, 1, 0); //
    f[1][1][0] = f_(1, 1, 0); //
    f[0][0][1] = f_(0, 0, 1); //
    f[1][0][1] = f_(1, 0, 1); //
    f[0][1][1] = f_(0, 1, 1); //
    f[1][1][1] = f_(1, 1, 1); //

    // std::array<std::function<double(double)>, 2> L = {{
    //     [](double x) { return (1.0-x); },
    //     [](double x) { return (x); }
    // }};

    double sum = 0.0;
    double x   = 0.2;
    double y   = 0.3;
    double z   = 0.3;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                sum += f[i][j][k] * L[i](x) * L[j](y) * L[k](z);
            }

    std::cout << sum << std::endl;
    std::cout << f_(x, y, z) << std::endl;

    return 0;
}
