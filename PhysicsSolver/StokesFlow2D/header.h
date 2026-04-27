/// @date 2023-06-06
/// @file header.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
/// 验证二维流体求解器的程序
///

#ifndef __HEADER_H__
#define __HEADER_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// using namespace std;

using std::exp;
using std::pow;

// const double EPSILON = 1e-15;
const double EPSILON = std::sqrt(1e-15);
// const double EPSILON = std::sqrt(std::numeric_limits<double>::epsilon());
// const double EPSILON = std::numeric_limits<double>::epsilon();

template <typename T>
auto make_vector_2D(int N, int M) {
    std::vector<std::vector<T>> result;
    result.resize(N, std::vector<T>(M));
    return result;
}

template <typename T>
double squared_distance(const T& a, const T& b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += squared_distance(a[i], b[i]);
    }
    // printf("%f\n", sum);
    return sum;
}

template <>
double squared_distance<double>(const double& a, const double& b) {
    return (a - b) * (a - b);
}

#endif