/// @date 2023-09-20
/// @file a.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <iostream>

template <int N, typename T>
void outer_product(const T* v1, T* v2, T* out) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            out[N * j + i] = v1[i] * v2[j];
        }
}

template <typename T>
T phi(T a) {
    return a;
}

template <int N, typename T>
void delta_function_weights(T r1, T r2, T* out) {
    double v1[N] = {-1 - r1, -r1, 1 - r1, 2 - r1};
    double v2[N] = {-1 - r2, -r2, 1 - r2, 2 - r2};

    for (size_t i = 0; i < N; i++) {
        v1[i] = phi(v1[i]);
        v2[i] = phi(v2[i]);
    }
    outer_product<N>(v1, v2, out);
}

int main() {
    double a[] = {1, 2};
    double b[] = {1, 2};
    double c[] = {1, 2, 3, 4};
    outer_product<2>(a, b, c);

    double out[16];
    delta_function_weights<4>(2.2, 3.3, c);
    return 0;
}
