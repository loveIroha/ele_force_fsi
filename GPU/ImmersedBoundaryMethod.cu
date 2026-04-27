/**
 * @file ImmersedBoundaryMethod.cu
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-07
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 * @updated 2023-04-17 插值算子和延拓算子考虑交错网格的半个网格步长
 * 
 */

#include <GPU/ImmersedBoundaryMethod.h>

namespace gpu
{
    template<typename T>
    __forceinline__ __device__ T ibm_phi(T r)
    {
        T res = 0;
        T r2 = r * r;
        if (r <= 2)
            res = 0.125 * (5 - 2 * r - sqrt(-7 + 12 * r - 4 * r2));
        if (r <= 1)
            res = 0.125 * (3 - 2 * r + sqrt(1 + 4 * r - 4 * r2));
        return res;
    }

    template<typename TV1, typename TV2>
    __forceinline__ __device__ double ibm_delta3(const TV1& x, const TV2& g)
    {

        return ibm_phi(abs(x.x - g.x)) * ibm_phi(abs(x.y - g.y)) * ibm_phi(abs(x.z - g.z));
    }

    __forceinline__ __device__ void distribute_force_single(
        const double3 &f,
        const double4 &quadrature_rule, // 积分点
        double3 *fluid_forces,
        double h, int3 dim, int index)
    {
        int i = floor(quadrature_rule.x / h);
        int j = floor(quadrature_rule.y / h);
        int k = floor(quadrature_rule.z / h);
        double w = quadrature_rule.w;
        double inv_h3 = 1.0 / h / h / h;

        for (int kk = k - 1; kk <= k + 2; kk++)
        {
            for (int jj = j - 1; jj <= j + 2; jj++)
            {
                for (int ii = i - 1; ii <= i + 2; ii++)
                {
                    if (!(kk >= 0 && kk < dim.z && jj >= 0 && jj < dim.y && ii >= 0 && ii < dim.x))
                    {
                        continue;
                    }
                    int widx = ii + jj * dim.x + kk * dim.x * dim.y;
                    double3 spreadf;
                    double weight = w * inv_h3 * ibm_delta3(quadrature_rule/h, make_double3(ii, jj, kk));
                    ((double *)&spreadf)[index] = ((double *)&f)[index] * weight;
                    atomicAdd(&(((double *)&fluid_forces[widx])[index]), ((double *)&spreadf)[index]);
                }
            }
        }
    }

    /// points Spread force from solid points to grid points
    /// inspired by xinxin's immersed boudnary method
    __global__ void distribute_force_kernel(
        const double3 *solid_forces,     /// solid_forces
        const double4 *quadrature_rules, /// quadrature quadrature_rules
        double3 *fluid_forces,
        int num, double h, int3 dim)
    {
        int gidx = blockIdx.x * blockDim.x + threadIdx.x;

        if (gidx < num)
        {
            double4 quadrature_rule = quadrature_rules[gidx] + double4{-0.5*h,-0.5*h,-0.5*h,0.0};
            double3 f = solid_forces[gidx];

            quadrature_rule.x += 0.5 * h;
            distribute_force_single(f, quadrature_rule, fluid_forces, h, dim, 0);
            quadrature_rule.x -= 0.5 * h;
            quadrature_rule.y += 0.5 * h;
            distribute_force_single(f, quadrature_rule, fluid_forces, h, dim, 1);
            quadrature_rule.y -= 0.5 * h;
            quadrature_rule.z += 0.5 * h;
            distribute_force_single(f, quadrature_rule, fluid_forces, h, dim, 2);
        }
    }

    __forceinline__ __device__ double interpolate_velocity_single(
        const double3 *fluid_velocities,
        const double4 &quadrature_rule, // 积分点
        double h, int3 dim, int index)
    {
        int i = floor(quadrature_rule.x / h);
        int j = floor(quadrature_rule.y / h);
        int k = floor(quadrature_rule.z / h);
        // double3 sum = make_double3(0, 0, 0);
        double sum = 0.0;
        for (int kk = k - 1; kk <= k + 2; kk++)
        {
            for (int jj = j - 1; jj <= j + 2; jj++)
            {
                for (int ii = i - 1; ii <= i + 2; ii++)
                {
                    if (!(kk >= 0 && kk < dim.z && jj >= 0 && jj < dim.y && ii >= 0 && ii < dim.x))
                    {
                        continue;
                    }
                    int ridx = ii + jj * dim.x + kk * dim.x * dim.y;
                    double weight = ibm_delta3(quadrature_rule / h, make_double3(ii, jj, kk));
                    double3 gvalue = fluid_velocities[ridx];
                    sum += weight * ((double *)&gvalue)[index];
                }
            }
        }
        return sum;
    }

    __global__ void interpolate_velocity_kernel(
        double3 *solid_velocities, const double4 *quadrature_rules, const double3 *fluid_velocities,
        int num, double h, int3 dim)
    {
        int gidx = blockIdx.x * blockDim.x + threadIdx.x;
        if (gidx < num)
        {
            double4 quadrature_rule = quadrature_rules[gidx] + double4{-0.5*h,-0.5*h,-0.5*h,0.0};
            quadrature_rule.x += 0.5 * h;
            solid_velocities[gidx].x = interpolate_velocity_single(fluid_velocities,quadrature_rule,h,dim,0);
            quadrature_rule.x -= 0.5 * h;
            quadrature_rule.y += 0.5 * h;
            solid_velocities[gidx].y = interpolate_velocity_single(fluid_velocities,quadrature_rule,h,dim,1);
            quadrature_rule.y -= 0.5 * h;
            quadrature_rule.z += 0.5 * h;
            solid_velocities[gidx].z = interpolate_velocity_single(fluid_velocities,quadrature_rule,h,dim,2);
        }
    }

    /// GPU kernel function for distribution operator
    template<typename TV, typename TV4, typename T>
    __global__ void distribute_center_kernel(
        const TV* solid_forces, const TV4* quadrature_rules, TV* fluid_forces,
        int num, T h, int3 dim)
    {
        constexpr int TV_size = sizeof(TV)/sizeof(T);

        int gidx = blockIdx.x * blockDim.x + threadIdx.x;
        if(gidx<num)
        {
            TV4 quadrature_rule = quadrature_rules[gidx];
            TV  f = solid_forces[gidx];
            T   w = quadrature_rule.w;
            T   inv_h3 = 1.0/h/h/h;
            
            int i = floor(quadrature_rule.x/h);
            int j = floor(quadrature_rule.y/h);
            int k = floor(quadrature_rule.z/h);


            for(int kk=k-1;kk<=k+2;kk++)
            {
                for(int jj=j-1;jj<=j+2;jj++)
                {
                    for(int ii=i-1;ii<=i+2;ii++)
                    {
                        if(!(kk>=0&&kk<dim.z&&jj>=0&&jj<dim.y&&ii>=0&&ii<dim.x))
                        {
                            continue;
                        }
                        int widx = ii + jj*dim.x + kk*dim.x*dim.y;
                        T weight = w*inv_h3*ibm_delta3(quadrature_rule/h, make_double3(ii,jj,kk));
                        
                        for (int ll = 0; ll < TV_size; ll++)
                        {
                            // f.x : ((double*)(&f))[ll]
                            // fluid_forces[widx].x : ((double*)(&(fluid_forces[widx])))[ll]
                            // spreadf.x = f.x*weight;
                            // atomicAdd(&(fluid_forces[widx].x), (spreadf.x));

                            atomicAdd(&(((double*)(&(fluid_forces[widx])))[ll]), ((double*)(&f))[ll]*weight);
                        }

                        // double3 spreadf;
                        // spreadf.x = f.x*weight;
                        // spreadf.y = f.y*weight;
                        // spreadf.z = f.z*weight;
                        // atomicAdd(&(fluid_forces[widx].x), (spreadf.x));
                        // atomicAdd(&(fluid_forces[widx].y), (spreadf.y));
                        // atomicAdd(&(fluid_forces[widx].z), (spreadf.z));
                        
                    }
                }
            }
        }
    }

    void distribute_force_gpu(
        const double3 *solid_forces, const double4 *quadrature_rules, double3 *fluid_forces,
        size_t num, double h, int3 dim)
    {
        std::cout << "\n\n call kernel distribute_force.\n\n"
                  << std::endl;
        uint numThreads, numBlocks;
        computeGridSize(num, 256, numBlocks, numThreads);
        distribute_force_kernel<<<numBlocks, numThreads>>>(solid_forces, quadrature_rules, fluid_forces, num, h, dim);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    void interpolate_velocity_gpu(
        double3 *solid_velocities, const double4 *quadrature_rules, const double3 *fluid_velocities,
        size_t num, double h, int3 dim)
    {
        std::cout << "\n\n call function interpolate_velocity_gpu:\n\n"
                  << std::endl;
        uint numThreads, numBlocks;
        computeGridSize(num, 256, numBlocks, numThreads);
        interpolate_velocity_kernel<<<numBlocks, numThreads>>>(solid_velocities, quadrature_rules, fluid_velocities, num, h, dim);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    void distribute_force(
        const double3 *solid_forces, const double4 *quadrature_rules, double3 *fluid_forces,
        size_t num, double h, int3 dim, bool useCUDA)
    {
        if (useCUDA)
        {
            cudaDeviceSynchronize();
            IBTimer timer("distribute_force with cuda.");

            double3 *solid_forces_dev;
            double4 *quadrature_rules_dev;
            double3 *fluid_forces_dev;

            // Malloc memory
            checkCudaErrors(cudaMalloc((void **)&solid_forces_dev, num * sizeof(double3)));
            checkCudaErrors(cudaMalloc((void **)&quadrature_rules_dev, num * sizeof(double4)));
            checkCudaErrors(cudaMalloc((void **)&fluid_forces_dev, dim.x * dim.y * dim.z * sizeof(double3)));

            // Copy
            checkCudaErrors(cudaMemcpy(solid_forces_dev, solid_forces, num * sizeof(double3), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(quadrature_rules_dev, quadrature_rules, num * sizeof(double4), cudaMemcpyHostToDevice));

            // Call
            distribute_force_gpu(solid_forces_dev, quadrature_rules_dev, fluid_forces_dev, num, h, dim);

            // Copy back the results
            checkCudaErrors(cudaMemcpy(fluid_forces, fluid_forces_dev, dim.x * dim.y * dim.z * sizeof(double3), cudaMemcpyDeviceToHost));

            // Free
            checkCudaErrors(cudaFree(solid_forces_dev));
            checkCudaErrors(cudaFree(quadrature_rules_dev));
            checkCudaErrors(cudaFree(fluid_forces_dev));
            cudaDeviceSynchronize();
        }
        else
        {
            cudaDeviceSynchronize();
            IBTimer timer("distribute_force without cuda.");
            distribute_force_cpu(solid_forces, quadrature_rules, fluid_forces, num, h, dim);
            cudaDeviceSynchronize();
        }
    }

    void interpolate_velocity(
        double3 *solid_velocities, const double4 *quadrature_rules, const double3 *fluid_velocities,
        size_t num, double h, int3 dim, bool useCUDA)
    {
        if (useCUDA)
        {
            cudaDeviceSynchronize();
            IBTimer timer("interpolate_velocity with cuda.");

            double3 *solid_velocities_dev;
            double4 *quadrature_rules_dev;
            double3 *fluid_velocities_dev;

            // Malloc memory
            checkCudaErrors(cudaMalloc((void **)&solid_velocities_dev, num * sizeof(double3)));
            checkCudaErrors(cudaMalloc((void **)&quadrature_rules_dev, num * sizeof(double4)));
            checkCudaErrors(cudaMalloc((void **)&fluid_velocities_dev, dim.x * dim.y * dim.z * sizeof(double3)));

            // Copy
            checkCudaErrors(cudaMemcpy(fluid_velocities_dev, fluid_velocities, dim.x * dim.y * dim.z * sizeof(double3), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(quadrature_rules_dev, quadrature_rules, num * sizeof(double4), cudaMemcpyHostToDevice));

            // Call
            interpolate_velocity_gpu(solid_velocities_dev, quadrature_rules_dev, fluid_velocities_dev, num, h, dim);

            // Copy back the results
            checkCudaErrors(cudaMemcpy(solid_velocities, solid_velocities_dev, num * sizeof(double3), cudaMemcpyDeviceToHost));

            // Free
            checkCudaErrors(cudaFree(solid_velocities_dev));
            checkCudaErrors(cudaFree(quadrature_rules_dev));
            checkCudaErrors(cudaFree(fluid_velocities_dev));
            cudaDeviceSynchronize();
        }
        else
        {
            cudaDeviceSynchronize();
            IBTimer timer("interpolate_velocity without cuda.");
            interpolate_velocity_cpu(solid_velocities, quadrature_rules, fluid_velocities, num, h, dim);
            cudaDeviceSynchronize();
        }
    }

    template<typename TV>
    void distribute_center_gpu(
        const TV *solid_forces, const double4 *quadrature_rules, TV *fluid_forces,
        size_t num, double h, int3 dim)
    {
        std::cout << "\n\n        distribute_center_gpu.\n\n" << std::endl;
        uint numThreads, numBlocks;
        computeGridSize(num, 256, numBlocks, numThreads);
        distribute_center_kernel<TV,double4,double><<<numBlocks, numThreads>>>(solid_forces, quadrature_rules, fluid_forces, num, h, dim);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

    template<typename TV>
    void distribute_center(
        const TV *solid_forces, const double4 *quadrature_rules, TV *fluid_forces,
        size_t num, double h, int3 dim, bool useCUDA)
    {
            cudaDeviceSynchronize();
            IBTimer timer("Distribute the scalar quantity defined at the center of cells using CUDA.");

            TV *solid_forces_dev;
            double4 *quadrature_rules_dev;
            TV *fluid_forces_dev;

            // Malloc memory
            checkCudaErrors(cudaMalloc((void **)&solid_forces_dev, num * sizeof(TV)));
            checkCudaErrors(cudaMalloc((void **)&quadrature_rules_dev, num * sizeof(double4)));
            checkCudaErrors(cudaMalloc((void **)&fluid_forces_dev, dim.x * dim.y * dim.z * sizeof(TV)));

            // Copy
            checkCudaErrors(cudaMemcpy(solid_forces_dev, solid_forces, num * sizeof(TV), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(quadrature_rules_dev, quadrature_rules, num * sizeof(double4), cudaMemcpyHostToDevice));

            // Call
            distribute_center_gpu<TV>(solid_forces_dev, quadrature_rules_dev, fluid_forces_dev, num, h, dim);

            // Copy back the results
            checkCudaErrors(cudaMemcpy(fluid_forces, fluid_forces_dev, dim.x * dim.y * dim.z * sizeof(TV), cudaMemcpyDeviceToHost));

            // Free
            checkCudaErrors(cudaFree(solid_forces_dev));
            checkCudaErrors(cudaFree(quadrature_rules_dev));
            checkCudaErrors(cudaFree(fluid_forces_dev));
            cudaDeviceSynchronize();
    }
    
    template void distribute_center<double3>(const double3 *solid_forces,const double4 *quadrature_rules,double3 *fluid_forces,size_t num, double h, int3 dim, bool useCUDA);
    template void distribute_center<double>(const double *solid_forces,const double4 *quadrature_rules,double *fluid_forces,size_t num, double h, int3 dim, bool useCUDA);
    
}