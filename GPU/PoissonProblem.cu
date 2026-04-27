/**
 * @file PoissonProblem.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-02-18
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */

#include <GPU/utilities.h>

// const char INTERIOR=0;
const char DIRICHLET=1;
// const char NEUMANN=2;

namespace gpu {
namespace poisson{


    template<typename T, typename TV> __global__
    void smooth_kernel(TV* _x, const TV* _b, int color, int3 dim, double3 dh){
        // CHECK_F(false, "Boundary conditions are not given.");
    }


    template<typename T, typename TV>
    void smooth(TV* x, const TV* b, int3 dim, double3 dh){
        // CHECK_F(false, "Boundary conditions are not given.");


    }

    template void smooth<double, double3>(double3* x, const double3* b, int3 dim, double3 dh);

    template<typename T, typename TV, typename BC> __global__
    void smooth_kernel(TV* _x, const TV* _b, const BC* _bc, int color, int3 dim, double3 dh){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if( i >=0 && i<dim.x && j>=0 && j<dim.y) {
            for (int k = 0; k < dim.z; k++)
            {
                auto centerindex = i + j*dim.x + k*dim.x*dim.y;
                
                if ((centerindex + color)%2==0)
                // {
                    
                //     if (bc[centerindex] == DIRICHLET) continue;

                //     auto xstride = 1;
                //     auto ystride = dim.x;
                //     auto zstride = dim.x*dim.y;

                //     auto l = i-1<0       ? _x[centerindex+xstride] : _x[centerindex-xstride];
                //     auto r = i+1>dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                //     auto d = j-1<0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                //     auto u = j+1>dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                //     auto b = k-1<0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                //     auto f = k+1>dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];


                //     T dx2 = dh.x*dh.x;
                //     T dy2 = dh.y*dh.y;
                //     T dz2 = dh.z*dh.z;

                //     _x[centerindex] = ((l+r)*dy2*dz2 + (u+d)*dx2*dz2 + (f+b)*dx2*dy2-_b[centerindex]*dz2*dx2*dy2)/(2.0*(dx2*dy2+dy2*dz2+dx2*dz2));
                // }

                {
                    if (_bc[i+j*dim.x+k*dim.x*dim.y] == DIRICHLET) continue;
                    
                    auto l = i-1<0       ? _x[i+1+j*dim.x    +    k*dim.x*dim.y]:_x[i-1+j*dim.x    +    k*dim.x*dim.y];
                    auto r = i+1>dim.x-1 ? _x[i-1+j*dim.x    +    k*dim.x*dim.y]:_x[i+1+j*dim.x    +    k*dim.x*dim.y];
                    auto d = j-1<0       ? _x[i  +(j+1)*dim.x+    k*dim.x*dim.y]:_x[i  +(j-1)*dim.x+    k*dim.x*dim.y];
                    auto u = j+1>dim.y-1 ? _x[i  +(j-1)*dim.x+    k*dim.x*dim.y]:_x[i  +(j+1)*dim.x+    k*dim.x*dim.y];
                    auto b = k-1<0       ? _x[i  +    j*dim.x+(k+1)*dim.x*dim.y]:_x[i  +    j*dim.x+(k-1)*dim.x*dim.y];
                    auto f = k+1>dim.z-1 ? _x[i  +    j*dim.x+(k-1)*dim.x*dim.y]:_x[i  +    j*dim.x+(k+1)*dim.x*dim.y];

                    auto dx2 = dh.x*dh.x;
                    auto dy2 = dh.y*dh.y;
                    auto dz2 = dh.z*dh.z;

                    _x[i+j*dim.x+k*dim.x*dim.y] = ((l+r)*dy2*dz2 + (u+d)*dx2*dz2 + (f+b)*dx2*dy2-_b[i+j*dim.x+k*dim.x*dim.y]*dz2*dx2*dy2)/(2.0*(dx2*dy2+dy2*dz2+dx2*dz2));
                }
            }
        }
    }

    template<typename T, typename TV, typename BC>
    void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh){

        dim3 blockSize;
        blockSize.x = 32;
        blockSize.y = 32;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        int red = 0;
        int black = 1;

        cudaDeviceSynchronize();

        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, red, dim, dh);
        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, black, dim, dh);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void smooth<double, double3, char>(double3* x, const double3* b, const char* bc , int3 dim, double3 dh);

    template<typename T, typename TV> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, int3 dim, double3 dh){
        // CHECK_F(false, "Boundary conditions are not given.");
    }


    template<typename T, typename TV>
    void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh){
        // CHECK_F(false, "Boundary conditions are not given.");

    }

    template void compute_residuals<double, double3>(const double3* x, const double3* b, double3* r, int3 dim, double3 dh);

    template<typename T, typename TV, typename BC> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, const BC* _bc, int3 dim, double3 dh){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if( i >= 0 && i<dim.x && j>=0 && j<dim.y) {
            for (int k = 0; k < dim.z; k++)
            {
                // auto centerindex = i+j*dim.x+k*dim.x*dim.y;

                // if (bc[centerindex] == DIRICHLET) continue;

                // auto xstride = 1;
                // auto ystride = dim.x;
                // auto zstride = dim.x*dim.y;
                // auto m = _x[centerindex];
                
                // auto l = i-1<0       ? _x[centerindex+xstride] : _x[centerindex-xstride];
                // auto r = i+1>dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                // auto d = j-1<0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                // auto u = j+1>dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                // auto b = k-1<0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                // auto f = k+1>dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];

                // auto dx2 = dh.x*dh.x;                    
                // auto dy2 = dh.y*dh.y;
                // auto dz2 = dh.z*dh.z;

                // _r[centerindex] = _b[centerindex] - (l+r-2.0*m)/dx2 - (u+d-2.0*m)/dy2 - (b+f-2.0*m)/dz2;
                {
                    if (_bc[i+j*dim.x+k*dim.x*dim.y] == DIRICHLET) continue;

                    auto m = _x[i+j*dim.x+k*dim.x*dim.y];
                    auto l = i-1<0       ? _x[i+1+j*dim.x    +    k*dim.x*dim.y]:_x[i-1+j*dim.x    +    k*dim.x*dim.y];
                    auto r = i+1>dim.x-1 ? _x[i-1+j*dim.x    +    k*dim.x*dim.y]:_x[i+1+j*dim.x    +    k*dim.x*dim.y];
                    auto d = j-1<0       ? _x[i  +(j+1)*dim.x+    k*dim.x*dim.y]:_x[i  +(j-1)*dim.x+    k*dim.x*dim.y];
                    auto u = j+1>dim.y-1 ? _x[i  +(j-1)*dim.x+    k*dim.x*dim.y]:_x[i  +(j+1)*dim.x+    k*dim.x*dim.y];
                    auto b = k-1<0       ? _x[i  +    j*dim.x+(k+1)*dim.x*dim.y]:_x[i  +    j*dim.x+(k-1)*dim.x*dim.y];
                    auto f = k+1>dim.z-1 ? _x[i  +    j*dim.x+(k-1)*dim.x*dim.y]:_x[i  +    j*dim.x+(k+1)*dim.x*dim.y];

                    auto dx2 = dh.x*dh.x;                    
                    auto dy2 = dh.y*dh.y;
                    auto dz2 = dh.z*dh.z;

                    _r[i+j*dim.x+k*dim.x*dim.y] = _b[i+j*dim.x+k*dim.x*dim.y] - (l+r-2.0*m)/dx2 - (u+d-2.0*m)/dy2 - (b+f-2.0*m)/dz2;
                }
            }
        }
    }


    template<typename T, typename TV, typename BC>
    void compute_residuals(const TV* x, const TV* b, TV* r, const BC* bc, int3 dim, double3 dh){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_residuals_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, r, bc, dim, dh);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_residuals<double, double3, char>(const double3* x, const double3* b, double3* r, const char* bc, int3 dim, double3 dh);


}

}


