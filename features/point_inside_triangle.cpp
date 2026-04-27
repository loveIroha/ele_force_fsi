#include <cmath>
#include <iostream>

struct Point3 {
    double x, y, z;
};

double dot_product(Point3 a, Point3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Point3 cross_product(Point3 a, Point3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double triangle_area(Point3 a, Point3 b, Point3 c) {
    Point3 v1 = {b.x - a.x, b.y - a.y, b.z - a.z};
    Point3 v2 = {c.x - a.x, c.y - a.y, c.z - a.z};
    Point3 cp = cross_product(v1, v2);
    return std::sqrt(dot_product(cp, cp)) / 2.0;
}

bool point_in_triangle(Point3 p, Point3 a, Point3 b, Point3 c) {
    double s  = triangle_area(a, b, c);
    double s1 = triangle_area(p, a, b);
    double s2 = triangle_area(p, b, c);
    double s3 = triangle_area(p, c, a);
    return std::abs(s1 + s2 + s3 - s) < 1e-9;
}

int main() {
    Point3 a  = {0.0, 0.0, 0.0};
    Point3 b  = {1.0, 0.0, 0.0};
    Point3 c  = {0.0, 1.0, 0.0};
    Point3 p1 = {0.5, 0.5, 0.0};
    Point3 p2 = {1.0, 1.0, 0.0};
    std::cout << std::boolalpha << point_in_triangle(p1, a, b, c) << std::endl; // true
    std::cout << std::boolalpha << point_in_triangle(p2, a, b, c) << std::endl; // false
    return 0;
}
