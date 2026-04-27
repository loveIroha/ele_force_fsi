/// @date 2023-11-24
/// @file norm.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试范数
///
///

#include <AlgebraSolver/algebra.h>
#include <iomanip>
#include <config.h>

// norm, add, average 等测试


int test_average() {
    // 输入不同的向量
    std::vector<double>                           un_single = {-1.0, 2.0, 3.0};
    std::vector<std::vector<double>>              un_nested = {{-1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<std::vector<double>>> un_deeply_nested
        = {{{-1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
    std::vector<std::vector<double2>> un_deeply_2 = {{{-1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};

    CHECK_F(std::abs(algebra::average(un_single) - 1.3333333333333333e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::average(un_nested) - 3.1666666666666665e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::average(un_deeply_nested) - 4.2500000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::average(un_deeply_2) - 4.2500000000000000e+00) < NPUHEART_EPS);

    // std::cout << "average of un_single:        " << algebra::average(un_single) << std::endl;
    // std::cout << "average of un_nested:        " << algebra::average(un_nested) << std::endl;
    // std::cout << "average of un_deeply_nested: " << algebra::average(un_deeply_nested) << std::endl;
    // std::cout << "average of un_deeply_2:      " << algebra::average(un_deeply_2) << std::endl;

    return 0;
}

int test_norm() {
    // 输入不同的向量
    std::vector<double>                           un_single = {-1.0, 2.0, 3.0};
    std::vector<std::vector<double>>              un_nested = {{-1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<std::vector<double>>> un_deeply_nested
        = {{{-1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};

    // 将标量值加到每个元素上
    double scalarToAdd = 1.0;
    algebra::add(un_single, scalarToAdd);
    algebra::add(un_nested, scalarToAdd);
    algebra::add(un_deeply_nested, scalarToAdd);

    // 测试不同范数
    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::l1) - 7.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::l2) - 5.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::linf) - 4.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::l1) - 2.5000000000000000e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::l2) - 1.1618950038622250e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::linf) - 7.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::l1) - 4.2000000000000000e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::l2) - 1.6733200530681511e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::linf) - 9.0000000000000000e+00) < NPUHEART_EPS);

    // std::cout << "l1   norm of un_single:        " << algebra::norm(un_single, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_single:        " << algebra::norm(un_single, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_single:        " << algebra::norm(un_single, algebra::Norm::linf) << std::endl;

    // std::cout << "l1   norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::linf) << std::endl;

    // std::cout << "l1   norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::linf) << std::endl;

    return 0;
}

int main() {
    // 输入不同的向量
    std::vector<double>                           un_single = {-1.0, 2.0, 3.0};
    std::vector<std::vector<double>>              un_nested = {{-1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<std::vector<std::vector<double>>> un_deeply_nested
        = {{{-1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};

    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::l1) - 6.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::l2) - 3.7416573867739413e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_single, algebra::Norm::linf) - 3.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::l1) - 2.1000000000000000e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::l2) - 9.5393920141694561e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_nested, algebra::Norm::linf) - 6.0000000000000000e+00) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::l1) - 3.6000000000000000e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::l2) - 1.4282856857085701e+01) < NPUHEART_EPS);
    CHECK_F(std::abs(algebra::norm(un_deeply_nested, algebra::Norm::linf) - 8.0000000000000000e+00) < NPUHEART_EPS);

    // 测试不同范数
    // std::cout << "l1   norm of un_single:        " << algebra::norm(un_single, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_single:        " << algebra::norm(un_single, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_single:        " << algebra::norm(un_single, algebra::Norm::linf) << std::endl;

    // std::cout << "l1   norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_nested:        " << algebra::norm(un_nested, algebra::Norm::linf) << std::endl;

    // std::cout << "l1   norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::l1) << std::endl;
    // std::cout << "l2   norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::l2) << std::endl;
    // std::cout << "linf norm of un_deeply_nested: " << algebra::norm(un_deeply_nested, algebra::Norm::linf) << std::endl;

    test_norm();
    test_average();
    return 0;
}
