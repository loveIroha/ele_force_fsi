/**
 * @file StokesProblem.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-02-20
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */

#include <GPU/utilities.h>

[[maybe_unused]] const char INTERIOR=0;
[[maybe_unused]] const char DIRICHLET=1;
[[maybe_unused]] const char NEUMANN=2;

namespace gpu {
namespace stokesflow{
namespace pressure{

    template<typename T, typename TV, typename BC> __global__
    void smooth_kernel(TV* _x, const TV* _b, const BC* _bc, int color, int3 dim, double3 dh){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int xstride = 1;
        const int ystride = dim.x; 
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        // Dirichlet boundary conditions on all boundaries for the pressure
        if( i<dim.x && j<dim.y) {
            for (int k = 0; k < dim.z; k++) {
                auto centerindex = i*xstride + j*ystride + k*zstride;

                if ((centerindex + color)%2==0 || _bc[centerindex]==DIRICHLET) continue;

                TV l = i == 0       ? _x[centerindex+xstride] : _x[centerindex-xstride];                    
                TV r = i == dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                TV u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                TV d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                TV f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                TV b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];

                _x[centerindex] = ((l+r)*dy2*dz2 + (u+d)*dx2*dz2 + (f+b)*dx2*dy2-_b[i+j*dim.x+k*dim.x*dim.y]*dz2*dx2*dy2)/(2.0*(dx2*dy2+dy2*dz2+dx2*dz2));
            }
        }
    }

    template<typename T, typename TV, typename BC>
    void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        int red   = 0;
        int black = 1;

        cudaDeviceSynchronize();

        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, red, dim, dh);
        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, black, dim, dh);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void smooth<double, double, char>(double* x, const double* b, const char* bc, int3 dim, double3 dh);

    template<typename T, typename TV, typename BC> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, const BC* _bc, int3 dim, double3 dh){
        
        const int xstride = 1;
        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(i<dim.x && j<dim.y ) {
            for (int k = 0; k < dim.z; k++)
            {
                    auto centerindex = i*xstride + j*ystride + k*zstride;
                    if (_bc[centerindex] == DIRICHLET) continue;

                    auto l = i == 0       ? _x[centerindex+xstride] : _x[centerindex-xstride];
                    auto r = i == dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                    auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                    auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                    auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                    auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];
            
                    auto m = _x[centerindex];

                    _r[centerindex] = _b[centerindex] - (l+r-2.0*m)/dx2 - (u+d-2.0*m)/dy2 - (b+f-2.0*m)/dz2;
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

    template void compute_residuals<double, double, char>(const double* x, const double* b, double* r, const char* bc, int3 dim, double3 dh);

    template<typename T, typename TV> __global__
    void compute_b_kernel(T* _b, const TV* _u, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        auto xstride = 1;
        auto ystride = _dim.x;  
        auto zstride = _dim.x*_dim.y;

        if( i<_dim.x-1 && j<_dim.y-1 ) {
            for (int k = 0; k < _dim.z-1; k++)
            {
                auto centerindex = i*xstride+j*ystride+k*zstride;
                _b[centerindex] =  (  (_u[centerindex + xstride].x - _u[centerindex].x)/_dh.x 
                                    + (_u[centerindex + ystride].y - _u[centerindex].y)/_dh.y
                                    + (_u[centerindex + zstride].z - _u[centerindex].z)/_dh.z )/_dt;
            }
        }
    }

    template<typename T, typename TV>
    void compute_b(T* b, const TV* u, int3 _dim, double3 _dh, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(_dim.x, blockSize.x);
        gridSize.y = iDivUp(_dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_b_kernel <T, TV> <<< gridSize, blockSize >>> (b, u, _dim, _dh, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_b<double, double3>(double* b, const double3* u, int3 _dim, double3 _dh, double _dt);

    template<typename T, typename TV> __global__
    void compute_b_with_source_kernel(T* _b, const TV* _u, const T* _s, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        auto xstride = 1;
        auto ystride = _dim.x;  
        auto zstride = _dim.x*_dim.y;

        if( i<_dim.x-1 && j<_dim.y-1 ) {
            for (int k = 0; k < _dim.z-1; k++)
            {
                auto centerindex = i*xstride+j*ystride+k*zstride;
                _b[centerindex] =  (  (_u[centerindex + xstride].x - _u[centerindex].x)/_dh.x 
                                    + (_u[centerindex + ystride].y - _u[centerindex].y)/_dh.y
                                    + (_u[centerindex + zstride].z - _u[centerindex].z)/_dh.z - _s[centerindex] )/_dt;
            }
        }
    }

    template<typename T, typename TV>
    void compute_b_with_source(T* b, const TV* u, const T* s, int3 _dim, double3 _dh, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(_dim.x, blockSize.x);
        gridSize.y = iDivUp(_dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_b_with_source_kernel <T, TV> <<< gridSize, blockSize >>> (b, u, s, _dim, _dh, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_b_with_source<double, double3>(double* b, const double3* u, const double* s, int3 _dim, double3 _dh, double _dt);

}

namespace velocity{

    template<typename T, typename TV, typename BC> __global__
    void smooth_kernel(TV* _x, const TV* _b, const BC* _bc, int color, int3 dim, double3 dh, double _mu, double _dt){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int xstride = 1;
        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;
        
        if( i<dim.x && j<dim.y) {
            for (int k = 0; k < dim.z; k++){
                auto centerindex = i*xstride + j*ystride + k*zstride;
                if ((centerindex + color)%2==0 || _bc[centerindex]==DIRICHLET) continue;
                
                auto l = i == 0       ? _x[centerindex+xstride] : _x[centerindex-xstride];
                auto r = i == dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];

                _x[centerindex] = (_b[centerindex] + (l+r)*_mu/dh.x/dh.x  + (u+d)*_mu/dh.y/dh.y + (f+b)*_mu/dh.z/dh.z) 
                    / (1.0/_dt + _mu*2.0/dh.x/dh.x + _mu*2.0/dh.y/dh.y + _mu*2.0/dh.z/dh.z);
                
            }
        }   
    }

    template<typename T, typename TV, typename BC>
    void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh, double _mu, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        int red = 0;
        int black = 1;

        cudaDeviceSynchronize();

        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, red, dim, dh, _mu, _dt);
        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, bc, black, dim, dh, _mu, _dt);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void smooth<double, double3, char>(double3* x, const double3* b, const char* bc, int3 dim, double3 dh, double _mu, double _dt);

    template<typename T, typename TV, typename BC> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, const BC* _bc, int3 dim, double3 dh, double _mu, double _dt){
        
        const int xstride = 1;
        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        if( i<dim.x && j<dim.y ) {
            for (int k = 0; k < dim.z; k++){
                auto centerindex = i*xstride + j*ystride + k*zstride;
                if (_bc[centerindex] == DIRICHLET) continue;

                auto l = i == 0       ? _x[centerindex+xstride] : _x[centerindex-xstride];
                auto r = i == dim.x-1 ? _x[centerindex-xstride] : _x[centerindex+xstride];
                auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];
                
                _r[centerindex] = 
                    _b[centerindex] - _x[centerindex]/_dt
                    + (l+r - 2.0*_x[centerindex])*_mu/dh.x/dh.x 
                    + (u+d - 2.0*_x[centerindex])*_mu/dh.y/dh.y
                    + (f+b - 2.0*_x[centerindex])*_mu/dh.z/dh.z;
            }
        }
    }

    template<typename T, typename TV, typename BC>
    void compute_residuals(const TV* x, const TV* b, TV* r, const BC* bc, int3 dim, double3 dh, double _mu, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_residuals_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, r, bc, dim, dh, _mu, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_residuals<double, double3, char>(const double3* x, const double3* b, double3* r, const char* bc, int3 dim, double3 dh, double _mu, double _dt);

    template<typename T, typename TV> __global__
    void compute_b_kernel(TV* _b, const TV* _f, const TV* _un, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        if( i<_dim.x && j<_dim.y ) {
            for (int k = 0; k < _dim.z; k++)
            {
                auto centerindex = i + j*_dim.x + k*_dim.x*_dim.y;
                _b[centerindex] = _f[centerindex] + _un[centerindex]/_dt;
            }
            
        }
    }

    template<typename T, typename TV>
    void compute_b(TV* b, const TV* f, const TV* un, int3 _dim, double3 _dh, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(_dim.x, blockSize.x);
        gridSize.y = iDivUp(_dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_b_kernel <T, TV> <<< gridSize, blockSize >>> (b, f, un, _dim, _dh, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_b<double, double3>(double3* b, const double3* f, const double3* un, int3 _dim, double3 _dh, double _dt);

}

    template<typename T, typename TV, typename BC> __global__
    void correct_velocity_kernel(TV* _u, const TV* _u_, const T* _p, const BC* _bc, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        auto xstride = 1;
        auto ystride = _dim.x;
        auto zstride = _dim.x * _dim.y;

        if( i<_dim.x && j<_dim.y ) {
            for (int k = 0; k <_dim.z; k++)
            {
                auto centerindex = i * xstride + j * ystride + k * zstride;
                if (_bc[centerindex] == DIRICHLET) continue; // Dirichlet boundary conditions.

                auto m = _p[centerindex];
                T r,d,b;

                if (i == _dim.x - 1){   r = _p[centerindex - 2*xstride];} 
                else if (i == 0){       r = _p[centerindex + xstride];} 
                else {                  r = _p[centerindex - xstride];}

                if (j == _dim.y - 1){   d = _p[centerindex - 2*ystride];} 
                else if (j == 0){       d = _p[centerindex + ystride];} 
                else {                  d = _p[centerindex - ystride];}

                if (k == _dim.z - 1){   b = _p[centerindex - 2*zstride];} 
                else if (k == 0){       b = _p[centerindex + zstride];} 
                else {                  b = _p[centerindex - zstride];}

                // auto r = i == _dim.x - 1 ? _p[centerindex - 2*xstride] : _p[centerindex - xstride];
                // auto d = j == _dim.y - 1 ? _p[centerindex - 2*ystride] : _p[centerindex - ystride];
                // auto b = k == _dim.z - 1 ? _p[centerindex - 2*zstride] : _p[centerindex - zstride];
                _u[centerindex].x = _u_[centerindex].x - _dt * (m - r) / _dh.x;
                _u[centerindex].y = _u_[centerindex].y - _dt * (m - d) / _dh.y;
                _u[centerindex].z = _u_[centerindex].z - _dt * (m - b) / _dh.z;
            }
        }
    }

    template<typename T, typename TV, typename BC>
    void correct_velocity(TV* u, const TV* u_, const T* p, const BC* bc, int3 _dim, double3 _dh, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(_dim.x, blockSize.x);
        gridSize.y = iDivUp(_dim.y, blockSize.y);

        cudaDeviceSynchronize();
        correct_velocity_kernel <T, TV> <<< gridSize, blockSize >>> (u, u_, p, bc, _dim, _dh, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void correct_velocity<double, double3, char>(double3* _u, const double3* _u_, const double* _p, const char* _bc, int3 _dim, double3 _dh, double _dt);


}

}