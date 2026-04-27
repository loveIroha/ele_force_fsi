#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

// 模板元编程：根据维度生成数组类型
template <size_t DIM, typename T>
struct MultiArray {
    using type = std::vector<typename MultiArray<DIM - 1, T>::type>;
};

template <typename T>
struct MultiArray<1, T> {
    using type = std::vector<T>;
};

// 递归创建多维数组
template <size_t DIM, typename T>
typename MultiArray<DIM, T>::type create_multi_array(const std::array<size_t, DIM>& dimensions) {
    typename MultiArray<DIM, T>::type result;

    if constexpr (DIM == 1) {
        result.resize(dimensions[0]);
    } else {
        result.resize(dimensions[0]);
        std::array<size_t, DIM - 1> reducedDimensions;
        for (size_t i = 0; i < DIM - 1; ++i) {
            reducedDimensions[i] = dimensions[i + 1];
        }
        for (size_t i = 0; i < dimensions[0]; ++i) {
            result[i] = create_multi_array<DIM - 1, T>(reducedDimensions);
        }
    }

    return result;
}

// 打印多维数组
template <size_t DIM, typename T>
void print_multi_array(const typename MultiArray<DIM, T>::type& data, int width = 8) {
    if constexpr (DIM == 1) {
        for (const auto& element : data) {
            std::cout << std::setw(width) << element;
        }
    } else {
        for (const auto& plane : data) {
            print_multi_array<DIM - 1, T>(plane, width);
        }
    }
    std::cout << std::endl;
}

int main() {
    std::array<size_t, 1> dimensions_1D = {5};
    std::array<size_t, 2> dimensions_2D = {4, 3};
    std::array<size_t, 3> dimensions_3D = {4, 3, 2};

    auto array_1D = create_multi_array<1, int>(dimensions_1D);
    auto array_2D = create_multi_array<2, double>(dimensions_2D);
    auto array_3D = create_multi_array<3, float>(dimensions_3D);

    // 初始化示例数据
    int value = 1;
    for (size_t i = 0; i < dimensions_1D[0]; ++i) {
        array_1D[i] = value++;
    }

    value = 1;
    for (size_t i = 0; i < dimensions_2D[0]; ++i) {
        for (size_t j = 0; j < dimensions_2D[1]; ++j) {
            array_2D[i][j] = static_cast<double>(value++);
        }
    }

    value = 1;
    for (size_t i = 0; i < dimensions_3D[0]; ++i) {
        for (size_t j = 0; j < dimensions_3D[1]; ++j) {
            for (size_t k = 0; k < dimensions_3D[2]; ++k) {
                array_3D[i][j][k] = static_cast<float>(value++);
            }
        }
    }

    // 打印数组内容
    std::cout << "1D Array:" << std::endl;
    print_multi_array<1, int>(array_1D);

    std::cout << "2D Array:" << std::endl;
    print_multi_array<2, double>(array_2D);

    std::cout << "3D Array:" << std::endl;
    print_multi_array<3, float>(array_3D);

    return 0;
}
