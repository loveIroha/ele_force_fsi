
#include <cassert>
#include <iostream>
#include <vector>

template <typename T>
double squared_distance(const T& a, const T& b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += squared_distance(a[i], b[i]);
    }
    printf("%f\n", sum);
    return sum;
}

template <>
double squared_distance<double>(const double& a, const double& b) {
    return (a - b) * (a - b);
}

int main() {
    std::vector<double> a{1, 2, 3, 4};
    std::vector<double> b{0, 1, 2, 3};
    printf("%f\n\n", squared_distance(a, b));

    std::vector<std::vector<double>> c{{1, 2}, {3, 4}, {1, 2}};
    std::vector<std::vector<double>> d{{0, 1}, {3, 3}, {0, 1}};
    printf("%f\n\n", squared_distance(c, d));
}