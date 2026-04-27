#include <array>
#include <iostream>

template <int N>
std::array<int, N> arr{};

template <>
std::array<int, 5> arr<5> = {1, 2, 3, 4, 5};

template <>
std::array<int, 6> arr<6> = {1, 2, 3, 4, 5, 6};

int main() {
    std::cout << arr<5>[4] << std::endl;
    std::cout << arr<6>[5] << std::endl;
    return 0;
}
