#include <assert.h>

#include <iostream>

struct float2 {
    float x;
    float y;
};

template <typename T>
double to_double(T a) {
    assert(sizeof(T) == sizeof(double));
    double* c = (double*)&a;
    return *c;
}
template <typename T>
float2 to_float2(const T& a) {
    assert(sizeof(T) == sizeof(float2));
    const float2* c = (const float2*)&a;
    return *c;
}

void test_to_double() {
    double a = 1.500000;
    float2 b = to_float2(a);
    float  c = b.x;
    a        = to_double(b);
    b        = to_float2(a);
    a        = to_double(c);

    // printf("a:%.20f\n",a);
    // printf("b[0]:%.20f\n", b.x);
    // printf("b[1]:%.20f\n", b.y);
}

int main() {
    test_to_double();
    test_to_float2();
    return 0;
}
