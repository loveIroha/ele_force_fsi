/// @date 2023-09-28
/// @file MultiArray.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __MULTI_ARRAY_H__
#define __MULTI_ARRAY_H__

#include <vector_types.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
namespace algebra {

// 模板元编程：根据维度生成数组类型
template <int DIM, typename T>
struct MultiArray {
    using type = std::vector<typename MultiArray<DIM - 1, T>::type>;
};

template <typename T>
struct MultiArray<1, T> {
    using type = std::vector<T>;
};

template <int DIM, typename T>
using MultiArrayType = typename algebra::MultiArray<DIM, T>::type;

template <typename T>
using Array3D  = typename algebra::MultiArrayType<3, T>;
using Array3Di = Array3D<int>;
using Array3Df = Array3D<float>;
using Array3Dd = Array3D<double>;

template <typename T>
using Array2D  = typename algebra::MultiArrayType<2, T>;
using Array2Di = Array2D<int>;
using Array2Df = Array2D<float>;
using Array2Dd = Array2D<double>;

// 递归创建多维数组
template <int DIM, typename T>
MultiArrayType<DIM, T> create_multi_array(const std::array<int, DIM>& dimensions) {
    MultiArrayType<DIM, T> result;

    if constexpr (DIM == 1) {
        result.resize(dimensions[0]);
    } else {
        result.resize(dimensions[0]);
        std::array<int, DIM - 1> reducedDimensions;
        for (int i = 0; i < DIM - 1; ++i) {
            reducedDimensions[i] = dimensions[i + 1];
        }
        for (int i = 0; i < dimensions[0]; ++i) {
            result[i] = create_multi_array<DIM - 1, T>(reducedDimensions);
        }
    }

    return result;
}

template <typename T>
void print_multi_array_single(const T& element, std::ostream& out, int width = 8) {
    out << std::fixed << std::setprecision(10) << (double)element << " ";
}
void print_multi_array_single(const double3& element, std::ostream& out, int width = 8) {
    out << std::fixed << std::setprecision(10) << (double)element.x << " "
        // << (double)element.y << " "
        // << (double)element.z << " "
        ;
}

// 打印多维数组
template <int DIM, typename T>
void print_multi_array(const MultiArrayType<DIM, T>& data, std::ostream& out, int width = 8) {
    if constexpr (DIM == 1) {
        for (const auto& element : data) {
            print_multi_array_single(element, out, width);
            // out << std::fixed << std::setprecision(10) << (double)element <<
            // " ";
        }
    } else {
        for (const auto& plane : data) {
            print_multi_array<DIM - 1, T>(plane, out, width);
            out << std::endl; // 每个子数组之间换行
        }
    }
}

} // namespace algebra
#endif // __MULTI_ARRAY_H__