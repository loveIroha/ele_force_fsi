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

namespace gpu {
namespace navier_stokes_flow{

namespace pressure{

    template<typename T, typename TV> __global__
    void smooth_kernel(TV* _x, const TV* _b, int color, int3 dim, double3 dh){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        // Dirichlet boundary conditions on all boundaries for the pressure
        if( i>0 && i<dim.x-1 &&  j>0 && j<dim.y-1 ) {
            for (int k = 1; k < dim.z-1; k++) {
                auto centerindex = i + j*dim.x + k*dim.x*dim.y;

                if ((centerindex + color)%2==0){

                    TV l = i == 0       ? _x[centerindex+1]       : _x[centerindex-1];
                    TV r = i == dim.x-1 ? _x[centerindex-1]       : _x[centerindex+1];
                    TV u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                    TV d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                    TV f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                    TV b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];

                    _x[i+j*dim.x+k*dim.x*dim.y] = ((l+r)*dy2*dz2 + (u+d)*dx2*dz2 + (f+b)*dx2*dy2-_b[i+j*dim.x+k*dim.x*dim.y]*dz2*dx2*dy2)/(2.0*(dx2*dy2+dy2*dz2+dx2*dz2));
                }
            }
        }
    }

    template<typename T, typename TV>
    void smooth(TV* x, const TV* b, int3 dim, double3 dh){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        int red   = 0;
        int black = 1;

        cudaDeviceSynchronize();

        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, red, dim, dh);
        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, black, dim, dh);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void smooth<double, double>(double* x, const double* b, int3 dim, double3 dh);

    template<typename T, typename TV> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, int3 dim, double3 dh){
        
        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if( i>0 && i<dim.x-1 &&  j>0 && j<dim.y-1 ) {
            for (int k = 1; k < dim.z-1; k++)
            {
                    auto centerindex = i + j*dim.x + k*dim.x*dim.y;

                    auto l = i == 0       ? _x[centerindex+1] : _x[centerindex-1];
                    auto r = i == dim.x-1 ? _x[centerindex-1] : _x[centerindex+1];
                    auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                    auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                    auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                    auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];
            
                    auto m = _x[centerindex];

                    _r[centerindex] = _b[centerindex] - (l+r-2.0*m)/dx2 - (u+d-2.0*m)/dy2 - (b+f-2.0*m)/dz2;
            }
        }
    }

    template<typename T, typename TV>
    void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_residuals_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, r, dim, dh);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_residuals<double, double>(const double* x, const double* b, double* r, int3 dim, double3 dh);

    template<typename T, typename TV> __global__
    void compute_b_kernel(T* _b, const TV* _u, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        auto ystride = _dim.x;
        auto zstride = _dim.x*_dim.y;

        if( i<_dim.x-1 && j<_dim.y-1 ) {
            for (int k = 0; k < _dim.z-1; k++)
            {
                auto centerindex = i+j*_dim.x+k*_dim.x*_dim.y;

                _b[centerindex] =  ( (_u[centerindex + 1].x       - _u[centerindex].x)/_dh.x 
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

}

namespace velocity{


    template<typename T, typename TV> __device__
    void calculate_convection_u(
        TV& convection, 
        const TV& c,
        const TV& l,
        const TV& r,
        const TV& u,
        const TV& d,
        const TV& f,
        const TV& b,
        T w_01,
        T v__0,
        const double3& dh)
    {

        T u_u = 0.5*(c.x + c.x);
        T u_d = 0.5*(c.x + d.x);
        T u_l = 0.5*(c.x + l.x);
        T u_r = 0.5*(c.x + r.x);
        T u_f = 0.5*(c.x + f.x);
        T u_b = 0.5*(c.x + b.x);

        T w_u = 0.5*(w_01 + u.z);
        T w_d = 0.5*(l.z  + c.z);
        T v_f = 0.5*(l.y + f.y);
        T v_b = 0.5*(v__0 + d.y);

        convection.y = dh.x*dh.y*u_u*w_u - dh.x*dh.y*u_d*w_d + dh.z*dh.y*u_r*u_r - dh.z*dh.y*u_l*u_l + dh.x*dh.z*u_f*v_f - dh.x*dh.z*u_b*v_b;
    }

      
    template<typename T, typename TV> __device__
    void calculate_convection_v(
        TV& convection, 
        const TV& c,
        const TV& l,
        const TV& r,
        const TV& u,
        const TV& d,
        const TV& f,
        const TV& b,
        T w011,
        T u110,
        const double3& dh)
    {
        T v_u = 0.5*(c.y + u.y);
        T v_d = 0.5*(c.y + d.y);
        T v_l = 0.5*(c.y + l.y);
        T v_r = 0.5*(c.y + r.y);
        T v_f = 0.5*(c.y + f.y);
        T v_b = 0.5*(c.y + b.y);

        T w_u = 0.5*(u.z + w011);
        T w_d = 0.5*(f.z + c.z);
        T u_r = 0.5*(u110+ r.x);
        T u_l = 0.5*(f.x + c.x);

        convection.y = dh.x*dh.y*v_u*w_u - dh.x*dh.y*v_d*w_d + dh.z*dh.y*v_r*u_r - dh.z*dh.y*v_l*u_l + dh.x*dh.z*v_f*v_f - dh.x*dh.z*v_b*v_b;
    }

    template<typename T, typename TV> __device__
    void calculate_convection_w(
        TV& convection, 
        const TV& c,
        const TV& l,
        const TV& r,
        const TV& u,
        const TV& d,
        const TV& f,
        const TV& b,
        T u10_,
        T v0__,
        const double3& dh)
    {
        T w_u = 0.5*(c.z + u.z);
        T w_d = 0.5*(c.z + d.z);
        T w_l = 0.5*(c.z + l.z);
        T w_r = 0.5*(c.z + r.z);
        T w_f = 0.5*(c.z + f.z);
        T w_b = 0.5*(c.z + b.z);

        T u_l = 0.5*(c.x + d.x);
        T u_r = 0.5*(u10_ + r.x);
        T v_f = 0.5*(c.y + d.y);
        T v_b = 0.5*(v0__ + b.y);
        
        convection.z = dh.x*dh.y*w_u*w_u - dh.x*dh.y*w_d*w_d + dh.z*dh.y*w_r*u_r - dh.z*dh.y*w_l*u_l + dh.x*dh.z*w_f*v_f - dh.x*dh.z*w_b*v_b;
    }
    
    template __device__ void calculate_convection_u<double, double3>(
        double3& convection, 
        const double3& c,
        const double3& l,
        const double3& r,
        const double3& u,
        const double3& d,
        const double3& f,
        const double3& b,
        double w_01,
        double v__0,
        const double3& dh);
    
    template __device__ void calculate_convection_v<double, double3>(
        double3& convection, 
        const double3& c,
        const double3& l,
        const double3& r,
        const double3& u,
        const double3& d,
        const double3& f,
        const double3& b,
        double w011,
        double u110,
        const double3& dh);
    
    template __device__ void calculate_convection_w<double, double3>(
        double3& convection, 
        const double3& c,
        const double3& l,
        const double3& r,
        const double3& u,
        const double3& d,
        const double3& f,
        const double3& b,
        double u10_,
        double v0__,
        const double3& dh);

    template<typename T, typename TV> __global__
    void smooth_kernel(TV* _x, const TV* _b, int color, int3 dim, double3 dh, double _mu, double _dt){

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;
        
        if( i > 0 && i<dim.x-1 && j>0 && j<dim.y-1 ) {
            for (int k = 1; k < dim.z-1; k++){
                
                auto centerindex = i + j*dim.x + k*dim.x*dim.y;
                if ((centerindex + color)%2==0){
                auto c = _x[centerindex];
                auto l = i == 0       ? _x[centerindex+1] : _x[centerindex-1];
                auto r = i == dim.x-1 ? _x[centerindex-1] : _x[centerindex+1];
                auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];

                auto x_01 = (i == 0                       || k == dim.z-1)  ? _x[centerindex-1        +zstride] : _x[centerindex];
                auto x__0 = (i == 0       || j == 0                      )  ? _x[centerindex-1-ystride        ] : _x[centerindex];
                auto x10_ = (i == dim.x-1                 || k == 0      )  ? _x[centerindex+1        -zstride] : _x[centerindex];
                auto x0__ = (                j == 0       || k == 0      )  ? _x[centerindex  -ystride-zstride] : _x[centerindex];
                auto x011 = (                j == dim.y-1 || k == dim.z-1)  ? _x[centerindex  +ystride+zstride] : _x[centerindex];
                auto x110 = (i == dim.x-1 || j == dim.y-1                )  ? _x[centerindex+1+ystride]         : _x[centerindex];
    
                auto w_01 = x_01.z;
                auto v__0 = x__0.y;
                auto u10_ = x10_.x;
                auto v0__ = x0__.y;        
                auto w011 = x011.z;
                auto u110 = x110.x;
                
                TV convection;
                calculate_convection_u<T, TV>(convection, c, l, r, u, d, f, b, w_01, v__0, dh);
                calculate_convection_v<T, TV>(convection, c, l, r, u, d, f, b, w011, u110, dh);
                calculate_convection_w<T, TV>(convection, c, l, r, u, d, f, b, u10_, v0__, dh);


                _x[centerindex] = (_b[centerindex] /*+ convection*/ + (l+r)*_mu/dh.x/dh.x  + (u+d)*_mu/dh.y/dh.y + (f+b)*_mu/dh.z/dh.z) 
                    / (1.0/_dt + _mu*2.0/dh.x/dh.x + _mu*2.0/dh.y/dh.y + _mu*2.0/dh.z/dh.z) ;
                }
            }
        }   
    }

    template<typename T, typename TV>
    void smooth(TV* x, const TV* b, int3 dim, double3 dh, double _mu, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        int red = 0;
        int black = 1;

        cudaDeviceSynchronize();

        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, red, dim, dh, _mu, _dt);
        smooth_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, black, dim, dh, _mu, _dt);

        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void smooth<double, double3>(double3* x, const double3* b, int3 dim, double3 dh, double _mu, double _dt);

    template<typename T, typename TV> __global__
    void compute_residuals_kernel(const TV* _x, const TV* _b, TV* _r, int3 dim, double3 dh, double _mu, double _dt){
        
        const int ystride = dim.x;
        const int zstride = dim.x*dim.y;

        const T dx2 = dh.x*dh.x;
        const T dy2 = dh.y*dh.y;
        const T dz2 = dh.z*dh.z;

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        if( i > 0 && i<dim.x-1 && j>0 && j<dim.y-1 ) {
            for (int k = 1; k < dim.z-1; k++){
                auto centerindex = i + j*dim.x + k*dim.x*dim.y;

                auto c = _x[centerindex];
                auto l = i == 0       ? _x[centerindex+1] : _x[centerindex-1];
                auto r = i == dim.x-1 ? _x[centerindex-1] : _x[centerindex+1];
                auto u = j == 0       ? _x[centerindex+ystride] : _x[centerindex-ystride];
                auto d = j == dim.y-1 ? _x[centerindex-ystride] : _x[centerindex+ystride];
                auto f = k == 0       ? _x[centerindex+zstride] : _x[centerindex-zstride];
                auto b = k == dim.z-1 ? _x[centerindex-zstride] : _x[centerindex+zstride];
                
                auto x_01 = (i == 0                       || k == dim.z-1)  ? _x[centerindex-1        +zstride] : _x[centerindex];
                auto x__0 = (i == 0       || j == 0                      )  ? _x[centerindex-1-ystride        ] : _x[centerindex];
                auto x10_ = (i == dim.x-1                 || k == 0      )  ? _x[centerindex+1        -zstride] : _x[centerindex];
                auto x0__ = (                j == 0       || k == 0      )  ? _x[centerindex  -ystride-zstride] : _x[centerindex];
                auto x011 = (                j == dim.y-1 || k == dim.z-1)  ? _x[centerindex  +ystride+zstride] : _x[centerindex];
                auto x110 = (i == dim.x-1 || j == dim.y-1                )  ? _x[centerindex+1+ystride]         : _x[centerindex];
    
                auto w_01 = x_01.z;
                auto v__0 = x__0.y;
                auto u10_ = x10_.x;
                auto v0__ = x0__.y;        
                auto w011 = x011.z;
                auto u110 = x110.x;
                
                TV convection;
                calculate_convection_u<T, TV>(convection, c, l, r, u, d, f, b, w_01, v__0, dh);
                calculate_convection_v<T, TV>(convection, c, l, r, u, d, f, b, w011, u110, dh);
                calculate_convection_w<T, TV>(convection, c, l, r, u, d, f, b, u10_, v0__, dh);

                _r[centerindex] = 
                    _b[centerindex] - c/_dt
                    + (l+r - 2.0*c)*_mu/dh.x/dh.x 
                    + (u+d - 2.0*c)*_mu/dh.y/dh.y
                    + (f+b - 2.0*c)*_mu/dh.z/dh.z
                    /*+ convection*/;
            }
        }
    }

    template<typename T, typename TV>
    void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh, double _mu, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(dim.x, blockSize.x);
        gridSize.y = iDivUp(dim.y, blockSize.y);

        cudaDeviceSynchronize();
        compute_residuals_kernel <T, TV> <<< gridSize, blockSize >>> (x, b, r, dim, dh, _mu, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void compute_residuals<double, double3>(const double3* x, const double3* b, double3* r, int3 dim, double3 dh, double _mu, double _dt);

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

    template<typename T, typename TV> __global__
    void correct_velocity_kernel(TV* _u, const TV* _u_, const T* _p, int3 _dim, double3 _dh, double _dt){
        
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int j = blockIdx.y * blockDim.y + threadIdx.y;

        if( i>0 && i<_dim.x-1 && j>0 && j<_dim.y-1 ) {
            for (int k = 1; k < _dim.z-1; k++)
            {
                auto centerindex = i+j*_dim.x+k*_dim.x*_dim.y;
                _u[centerindex].x = _u_[centerindex].x - _dt*(_p[centerindex] - _p[centerindex - 1])/_dh.x;
                _u[centerindex].y = _u_[centerindex].y - _dt*(_p[centerindex] - _p[centerindex - _dim.x])/_dh.y;
                _u[centerindex].z = _u_[centerindex].z - _dt*(_p[centerindex] - _p[centerindex - _dim.x*_dim.y])/_dh.z;
            }
            
        }
    }

    template<typename T, typename TV>
    void correct_velocity(TV* u, const TV* u_, const T* p, int3 _dim, double3 _dh, double _dt){

        dim3 blockSize;
        blockSize.x = 16;
        blockSize.y = 16;
        dim3 gridSize;
        gridSize.x = iDivUp(_dim.x, blockSize.x);
        gridSize.y = iDivUp(_dim.y, blockSize.y);

        cudaDeviceSynchronize();
        correct_velocity_kernel <T, TV> <<< gridSize, blockSize >>> (u, u_, p, _dim, _dh, _dt);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template void correct_velocity<double, double3>(double3* _u, const double3* _u_, const double* _p, int3 _dim, double3 _dh, double _dt);


}

}