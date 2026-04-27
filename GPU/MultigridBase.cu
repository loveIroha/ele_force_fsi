/**
 * @file MultigridBase.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-02-18
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */

#include <GPU/utilities.h>


namespace gpu {

    // TODO : Why is prolongation is much more expensive than restriction.
    template<typename T, typename TV> __global__
    void prolongate3D_kernel(TV* f, const TV* c, int3 dim_fine, int3 dim_coarse){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        auto ystride_coarse = dim_coarse.x;
        auto zstride_coarse = dim_coarse.x*dim_coarse.y;

        if(i<dim_fine.x && j<dim_fine.y){
            for (int k = 0; k < dim_fine.z; k++)
            {

                T u = 0.5*(i%2);
                T w = 0.5*(j%2);
                T v = 0.5*(k%2);

                int ii = i/2+1;
                int jj = j/2+1;
                int kk = k/2+1;

                auto centerindex = ii + jj*dim_coarse.x + kk*dim_coarse.x*dim_coarse.y;

                auto data0 = c[centerindex - 1 - ystride_coarse - zstride_coarse];
                auto data1 = ii > dim_coarse.x-1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - ystride_coarse - zstride_coarse];
                auto data2 = jj > dim_coarse.y-1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - 1 - zstride_coarse];
                auto data3 = (ii > dim_coarse.x-1 || jj > dim_coarse.y-1) ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - zstride_coarse];

                auto data4 = kk > dim_coarse.z-1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - 1 - ystride_coarse];
                auto data5 = (kk > dim_coarse.z-1 || ii > dim_coarse.x-1) ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - ystride_coarse];
                auto data6 = (kk > dim_coarse.z-1 || jj > dim_coarse.y-1) ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex - 1];
                auto data7 = (kk > dim_coarse.z-1 || ii > dim_coarse.x-1 || jj > dim_coarse.y-1) ? c[centerindex - 1 - ystride_coarse - zstride_coarse] : c[centerindex];

                T a[] = { (1.0-u)*(1.0-w)*(1.0-v), (1.0-w)*u*(1.0-v), (1.0-u)*w*(1.0-v),w*u*(1.0-v) };
                T b[] = { (1.0-u)*(1.0-w)*v,       (1.0-w)*u*v,       (1.0-u)*w*v,      w*u*v };
                
                TV e[] = { data0, data1, data2, data3 };
                TV d[] = { data4, data5, data6, data7 };

                // dot(e,a)+dot(d,b)
                f[i+j*dim_fine.x+k*dim_fine.x*dim_fine.y] = 
                    a[0]*e[0] + a[1]*e[1] + a[2]*e[2] + a[3]*e[3] +
                    b[0]*d[0] + b[1]*d[1] + b[2]*d[2] + b[3]*d[3];
            }
        }
    }

    

    template<typename T, typename TV> 
    void prolongate3D(TV* f, const TV* c, int3 dim_fine, int3 dim_coarse){

        dim3 blockSize;
        // TODO : When blockSize is (32,32), it doesn't work, but, why?
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim_fine.x, blockSize.x);
        gridSize.y = iDivUp(dim_fine.y, blockSize.y);

        cudaDeviceSynchronize();

        prolongate3D_kernel <T, TV> <<< gridSize, blockSize >>> (f, c, dim_fine, dim_coarse);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }


    template<typename T, typename TV> __global__
    void restrict3D_kernel(const TV* f, TV* c, int3 dim_fine, int3 dim_coarse){


        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        // auto xstride = 1;
        auto ystride = dim_fine.x;
        auto zstride = dim_fine.x*dim_fine.y;

        if(i<dim_coarse.x && j<dim_coarse.y){
            for (int k = 0; k < dim_coarse.z; k++){

                auto ii = 2*i;
                auto jj = 2*j;
                auto kk = 2*k;

                auto centerindex = ii + jj*ystride + kk*zstride;


                TV ll = (ii==0)                               ? f[centerindex+1]       : f[centerindex-1];
                TV rr = (ii==dim_fine.x - 1)                  ? f[centerindex-1]       : f[centerindex+1];
                TV dd = (jj==0)                               ? f[centerindex+ystride] : f[centerindex-ystride];
                TV uu = (jj==dim_fine.y - 1)                  ? f[centerindex-ystride] : f[centerindex+ystride];
                TV ff = (kk==0)                               ? f[centerindex+zstride] : f[centerindex-zstride];
                TV bb = (kk==dim_fine.z - 1)                  ? f[centerindex-zstride] : f[centerindex+zstride];
                TV mm = f[centerindex];
                c[i+j*dim_coarse.x+k*dim_coarse.x*dim_coarse.y] = (ll+rr+dd+uu+ff+bb)/12.0 + mm*0.5;


            }
        }
    

    }


    template<typename T, typename TV> 
    void restrict3D(const TV* f, TV* c, int3 dim_fine, int3 dim_coarse){

        // std::cout << "\n\n call kernel restrict3D.\n\n" << std::endl;
        
        dim3 blockSize;
        blockSize.x = 32;
        blockSize.y = 32;
        dim3 gridSize;
        gridSize.x = iDivUp(dim_coarse.x, blockSize.x);
        gridSize.y = iDivUp(dim_coarse.y, blockSize.y);

        cudaDeviceSynchronize();

        restrict3D_kernel <T, TV> <<< gridSize, blockSize >>> (f, c, dim_fine, dim_coarse);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }


    template void prolongate3D<double, double>(double* f, const double* c, int3 dim_fine, int3 dim_coarse);
    // template void prolongate3D<double, double2>(double2* f, const double2* c, int3 dim_fine, int3 dim_coarse);
    template void prolongate3D<double, double3>(double3* f, const double3* c, int3 dim_fine, int3 dim_coarse);

    // template void prolongate3D<float, float>(float* f, const float* c, int3 dim_fine, int3 dim_coarse);
    // template void prolongate3D<float, float2>(float2* f, const float2* c, int3 dim_fine, int3 dim_coarse);
    // template void prolongate3D<float, float3>(float3* f, const float3* c, int3 dim_fine, int3 dim_coarse);

    // template void restrict3D<float, float>(const float* f, float* c, int3 dim_fine, int3 dim_coarse);
    // template void restrict3D<float, float2>(const float2* f, float2* c, int3 dim_fine, int3 dim_coarse);
    // template void restrict3D<float, float3>(const float3* f, float3* c, int3 dim_fine, int3 dim_coarse);

    template void restrict3D<double, double>(const double* f, double* c, int3 dim_fine, int3 dim_coarse);
    // template void restrict3D<double, double2>(const double2* f, double2* c, int3 dim_fine, int3 dim_coarse);
    template void restrict3D<double, double3>(const double3* f, double3* c, int3 dim_fine, int3 dim_coarse);

}