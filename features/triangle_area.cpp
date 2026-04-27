#include <cmath>
#include <iostream>

struct double3 {
    double x, y, z;
};

double triangle_area(const double3* points) {
    double3 u   = {points[1].x - points[0].x, points[1].y - points[0].y, points[1].z - points[0].z};
    double3 v   = {points[2].x - points[0].x, points[2].y - points[0].y, points[2].z - points[0].z};
    double  det = u.x * v.y - u.y * v.x;
    return 0.5 * std::fabs(det);
}

int main() {
    double3 points[]      = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}};
    double  expected_area = 0.5;
    double  actual_area   = triangle_area(points);
    double  epsilon       = 1e-6;

    if (fabs(actual_area - expected_area) < epsilon) {
        std::cout << "Test passed: Triangle area is " << actual_area << std::endl;
    } else {
        std::cout << "Test failed: Expected " << expected_area << ", but got " << actual_area << std::endl;
    }

    return 0;
}
