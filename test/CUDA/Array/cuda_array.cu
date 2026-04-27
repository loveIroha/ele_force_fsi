#include <advection/CudaArray.cuh>
#include <memory>

__global__ void advect_u_kernel(
    CudaSurfaceAccessor<float2> surface_data,
    uint3 n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x == 0 || y == 0 || z == 0 || x >= n.x - 1 || y >= n.y - 1 || z >= n.z - 1)
        return;

    surface_data.write_double(x+1.0 / 3.0, x, y, z);
}

__global__ void advect_u_kernel_2(
    CudaSurfaceAccessor<float2> surface_data,
    uint3 n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x == 0 || y == 0 || z == 0 || x >= n.x - 1 || y >= n.y - 1 || z >= n.z - 1)
        return;

    double a = surface_data.read_double(x, y, z);
    float b = a;

    if (x == 5)
    {
        printf("a = %.16f\n", b);
    }
}

// struct Padding
// {
//     int left{0}, right{0}, top{0}, bottom{0};
// };


class CudaGridArray<T, DIM>
{
public:
    Padding padding;
    std::vector<T> data;
    std::array<int, 2> size_raw;
    const std::array<int, 2> size;

    GridArray(std::array<int, DIM> size) : size_raw(size), size(size)
    {
        data.resize(size[0] * size[1]);
    }
};

auto b_dim3(unsigned int Nx, unsigned int Ny, unsigned int Nz)
{
    return dim3((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);
}
int main()
{
    uint Nx = 100;
    uint Ny = 100;
    uint Nz = 100;
    double a = 1.0 / 3.0;
    float b = 1.0 / 3.0;
    printf("a = %.16f, b = %.16f\n", a, b);

    auto vel_u = std::make_unique<CudaSurface<float2>>(uint3{Nx + 1, Ny + 2, Nz + 2});

    advect_u_kernel<<<b_dim3(Nx, Ny, Nz), dim3(8, 8, 8)>>>(
        vel_u->accessSurface(),
        uint3{Nx, Ny, Nz});

    advect_u_kernel_2<<<b_dim3(Nx, Ny, Nz), dim3(8, 8, 8)>>>(
        vel_u->accessSurface(),
        uint3{Nx, Ny, Nz});
    return 0;
}
