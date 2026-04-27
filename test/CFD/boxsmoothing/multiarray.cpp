/// @date 2023-10-16
/// @file multiarray.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

struct double3 {
    double x, y, z;
};

// 模板元编程：根据维度生成数组类型
template <int DIM, typename T>
struct MultiArray {
    using type = std::vector<typename MultiArray<DIM - 1, T>::type>;
    type data;
};

template <typename T>
struct MultiArray<1, T> {
    using type = std::vector<T>;
    type data;
};

// 递归创建多维数组
template <int DIM = 3, typename T = double>
MultiArray<DIM, T> create_multi_array(const std::array<int, DIM>& dimensions) {
    MultiArray<DIM, T> result;

    if constexpr (DIM == 1) {
        result.data.resize(dimensions[0]);
    } else {
        result.data.resize(dimensions[0]);
        std::array<int, DIM - 1> reducedDimensions;
        for (int i = 0; i < DIM - 1; ++i) {
            reducedDimensions[i] = dimensions[i + 1];
        }
        for (int i = 0; i < dimensions[0]; ++i) {
            // OPTIMIZE: 能否使用move减少内存分配次数？
            auto temp      = create_multi_array<DIM - 1, T>(reducedDimensions);
            result.data[i] = temp.data;
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

template <int DIM, typename T>
using MultiArrayType = typename MultiArray<DIM, T>::type;

// 打印多维数组
template <int DIM, typename T>
void print_multi_array(const MultiArrayType<DIM, T>& data, std::ostream& out, int width = 8) {
    if constexpr (DIM == 1) {
        for (const auto& element : data) {
            print_multi_array_single(element, out, width);
            // out << std::fixed << std::setprecision(10) << (double)element << "
            // ";
        }
    } else {
        for (const auto& plane : data) {
            print_multi_array<DIM - 1, T>(plane, out, width);
            out << std::endl; // 每个子数组之间换行
        }
    }
}

int main() {
    auto multi_array = create_multi_array({2, 3, 4});
    print_multi_array<3, double>(multi_array.data, std::cout);
}