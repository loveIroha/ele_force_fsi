#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
auto generate_combinations(std::vector<T> element, int n) {
    auto getDigitAtPosition
        = [](int base, int num, int M) { return (num / static_cast<int>(std::pow(base, M))) % base; };

    // 生成所有可能的排列组合
    std::vector<std::vector<T>> combinations;
    for (int i = 0; i < std::pow(element.size(), n); ++i) {
        std::vector<T> combo;
        for (int j = 0; j < n; ++j) {
            int index = getDigitAtPosition(element.size(), i, n - j - 1);
            assert(index < element.size() && index >= 0);
            combo.push_back(element[index]);
        }
        combinations.push_back(combo);
    }
    return combinations;
};

int main() {
    int                      n       = 6;                                    // 元素列表的长度
    std::vector<std::string> element = {"NEUMANN", "DIRICHLET", "INTERIOR"}; // 元素列表的成员

    auto combinations = generate_combinations(element, n);

    // 输出所有排列组合
    for (const auto& combo : combinations) {
        for (auto num : combo) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
