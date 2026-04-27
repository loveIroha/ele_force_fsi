
#include <AlgebraSolver/GpuVector.h>
#include <GPU/gpu_lib.h>
#include <helper_math.h>
// standard library
#include <type_traits>

#define NEUMANN_TYPE 0
#define DIRICHLET_TYPE 1
#define MATERIAL_TYPE 2

int main() {
    int3                       dim = make_int3(100, 1, 1);
    GpuVector<double, double3> gpu_u(dim);
    StdVector<double, double3> _u(dim);

    int i = 2;
    int k = 2;
    int j = 2;

    using TV = std::decay_t<decltype(_u.data()[0])>;
    using T  = std::decay_t<decltype(flatten(_u).data[0])>;

    TV m = _u.data()[i + j * dim.x + k * dim.x * dim.y];
    TV l = _u.data()[i - 1 + j * dim.x + k * dim.x * dim.y];
    TV r = _u.data()[i + 1 + j * dim.x + k * dim.x * dim.y];
    TV u = _u.data()[i + (j + 1) * dim.x + k * dim.x * dim.y];
    TV d = _u.data()[i + (j - 1) * dim.x + k * dim.x * dim.y];
    TV f = _u.data()[i + j * dim.x + (k - 1) * dim.x * dim.y];
    TV b = _u.data()[i + j * dim.x + (k + 1) * dim.x * dim.y];

    m + l + r + u + d + f - b;

    // double3 diffusion =
    //     (_u[i+1 +     j*dim.x + k*dim.x*dim.y] - 2.0*_u[i + j*dim.x +
    //     k*dim.x*dim.y] + _u[i-1   + j*dim.x + k*dim.x*dim.y])/dh.x/dh.x +
    //     (_u[i   + (j+1)*dim.x + k*dim.x*dim.y] - 2.0*_u[i + j*dim.x +
    //     k*dim.x*dim.y] + _u[i + (j-1)*dim.x + k*dim.x*dim.y])/dh.y/dh.y +
    //     (_u[i+1 + j*dim.x + (k+1)*dim.x*dim.y] - 2.0*_u[i + j*dim.x +
    //     k*dim.x*dim.y] + _u[i + j*dim.x + (k-1)*dim.x*dim.y])/dh.z/dh.z
    // ;
    TV dh{0.1, 0.1, 0.1};
    T  nv = 0.1;

    TV diffusion
        = nv * ((l - m * 2.0 + r) / dh.x / dh.x + (u - m * 2.0 + d) / dh.y / dh.y + (f - 2.0 * m + b) / dh.z / dh.z);

    // 先处理边界条件，再进行迭代计算。

    // char3 boundary;
    // if sum(boundary) < 6
    //     handle boundary conditions

    // 关于NEUMANN边界条件有两种方法去处理。
    // (i,j,k)处是neumann边界条件，那么判断 (i,j,k-1) (i,j-1,k) (i-1,j,k)
    // 是否为内点，如果是内点，那么将它赋值给 (i,j,k)
    // (i,j,k)处是Dirichlet边界条件，那么直接给他赋值

    // 如果在赋值的时候进行计算

    return 0;
}