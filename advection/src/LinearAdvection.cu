/// @date 2023-10-30
/// @file LinearAdvection.cu
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief 2024-01-11 改变了虚拟点数量
/// 
///
#include <advection/LinearAdvection.h>

__global__ void advect_u_kernel(
    CudaTextureAccessor<float> texVel_u,
    CudaTextureAccessor<float> texVel_v,
    CudaTextureAccessor<float> texVel_w,
    CudaSurfaceAccessor<float> sufVel_u,
    uint3 n, float3 h, float dt)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x == 0 || y == 0 || z == 0 || x >= n.x - 1 || y >= n.y - 1 || z >= n.z - 1)
        return;

    float3 loc = make_float3(x - 1.0f, y - 0.5f, z - 0.5f);
    float3 vel = make_float3(
        texVel_u.sample(x + 0.5f, y + 0.5f, z + 0.5f),
        texVel_v.sample(x + 0.0f, y + 1.0f, z + 0.5f),
        texVel_w.sample(x + 0.0f, y + 0.5f, z + 1.0f));
    
    // if(x==2&&y==3&&z==4)
    //     printf("u:\nindex: %d %d %d\nlocation: %.20f %.20f %.20f\nvelocity: %.20f %.20f %.20f\n",
    //             x, y, z,
    //             loc.x, loc.y, loc.z,
    //             vel.x, vel.y, vel.z);

    loc.x -= vel.x / h.x * dt;
    loc.y -= vel.y / h.y * dt;
    loc.z -= vel.z / h.z * dt;

    float u_new = texVel_u.sample(loc.x + 1.5f, loc.y + 1.0f, loc.z + 1.0f);
    sufVel_u.write((u_new+vel.x)*0.5, x, y, z);
}

__global__ void advect_v_kernel(
    CudaTextureAccessor<float> texVel_u,
    CudaTextureAccessor<float> texVel_v,
    CudaTextureAccessor<float> texVel_w,
    CudaSurfaceAccessor<float> sufVel_v,
    uint3 n, float3 h, float dt)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x == 0 || y == 0 || z == 0 || x >= n.x - 1 || y >= n.y - 1 || z >= n.z - 1)
        return;

    float3 loc = make_float3(x - 0.5f, y - 1.0f, z - 0.5f);
    float3 vel = make_float3(
        texVel_u.sample(x + 1.0f, y + 0.0f, z + 0.5f),
        texVel_v.sample(x + 0.5f, y + 0.5f, z + 0.5f),
        texVel_w.sample(x + 0.5f, y + 0.0f, z + 1.0f));


    // if(x==2&&y==3&&z==4)
    //     printf("v:\nindex: %d %d %d\nlocation: %.20f %.20f %.20f\nvelocity: %.20f %.20f %.20f\n",
    //             x, y, z,
    //             loc.x, loc.y, loc.z,
    //             vel.x, vel.y, vel.z);


    loc.x -= vel.x / h.x * dt;
    loc.y -= vel.y / h.y * dt;
    loc.z -= vel.z / h.z * dt;

    float v_new = texVel_v.sample(loc.x + 1.0f, loc.y + 1.5f, loc.z + 1.0f);
    sufVel_v.write((v_new+vel.y)*0.5, x, y, z);
}

__global__ void advect_w_kernel(
    CudaTextureAccessor<float> texVel_u,
    CudaTextureAccessor<float> texVel_v,
    CudaTextureAccessor<float> texVel_w,
    CudaSurfaceAccessor<float> sufVel_w,
    uint3 n, float3 h, float dt)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x == 0 || y == 0 || z == 0 || x >= n.x - 1 || y >= n.y - 1 || z >= n.z - 1)
        return;

    float3 loc = make_float3(x - 0.5f, y - 0.5f, z - 1.0f);
    float3 vel = make_float3(
        texVel_u.sample(x + 1.0f, y + 0.5f, z + 0.0f),
        texVel_v.sample(x + 0.5f, y + 1.0f, z + 0.0f),
        texVel_w.sample(x + 0.5f, y + 0.5f, z + 0.5f));


    // if(x==2&&y==3&&z==4)
    //     printf("w:\nindex: %d %d %d\nlocation: %.20f %.20f %.20f\nvelocity: %.20f %.20f %.20f\n",
    //             x, y, z,
    //             loc.x, loc.y, loc.z,
    //             vel.x, vel.y, vel.z);


    loc.x -= vel.x / h.x * dt;
    loc.y -= vel.y / h.y * dt;
    loc.z -= vel.z / h.z * dt;

    float w_new = texVel_w.sample(loc.x + 1.0f, loc.y + 1.0f, loc.z + 1.5f);
    sufVel_w.write((w_new+vel.z)*0.5, x, y, z);

    // printf("w:\nindex: %f %f %f\nlocation: %.20f %.20f %.20f\nvelocity: %.20f %.20f %.20f %f\n",
    //        x - 0.5f, y - 0.5f, z,
    //        loc.x, loc.y, loc.z,
    //        vel.x, vel.y, vel.z,
    //        w_new);
}

void LinearAdvection::advection()

{
    advect_u_kernel<<<dim3((Nx + 3 + 7) / 8, (Ny + 2 + 7) / 8, (Nz + 2 + 7) / 8), dim3(8, 8, 8)>>>(
        vel_u->accessTexture(),
        vel_v->accessTexture(),
        vel_w->accessTexture(),
        vel_u_next->accessSurface(),
        uint3{Nx + 3, Ny + 2, Nz + 2}, make_float3(width / Nx, height / Ny, depth / Nz), ht); 

    advect_v_kernel<<<dim3((Nx + 2 + 7) / 8, (Ny + 3 + 7) / 8, (Nz + 2 + 7) / 8), dim3(8, 8, 8)>>>(
        vel_u->accessTexture(),
        vel_v->accessTexture(),
        vel_w->accessTexture(),
        vel_v_next->accessSurface(),
        uint3{Nx + 2, Ny + 3, Nz + 2}, make_float3(width / Nx, height / Ny, depth / Nz), ht);

    advect_w_kernel<<<dim3((Nx + 2 + 7) / 8, (Ny + 2 + 7) / 8, (Nz + 3 + 7) / 8), dim3(8, 8, 8)>>>(
        vel_u->accessTexture(),
        vel_v->accessTexture(),
        vel_w->accessTexture(),
        vel_w_next->accessSurface(),
        uint3{Nx + 2, Ny + 2, Nz + 3}, make_float3(width / Nx, height / Ny, depth / Nz), ht);

    // std::swap(vel_u, vel_u_next);
    // std::swap(vel_v, vel_v_next);
    // std::swap(vel_w, vel_w_next);
}