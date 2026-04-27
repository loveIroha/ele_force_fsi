/**
 * @file PressureSolverGPU.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-14
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PRESSURE_SOLVER_GPU_H__
#define __PRESSURE_SOLVER_GPU_H__

#include <GPU/gpu_lib.h>
#include <PhysicsSolver/multigrid/MultigridBase.h>

namespace pangu {

using PressureBCgpu = GpuVector<char, char>;

template <typename PressureType, typename VelocityType>
class PressureSolverGPU : public MultigridBase<PressureType, PressureBCgpu, DIM3> {
    using MultigridBase<PressureType, PressureBCgpu, DIM3>::_dim;
    using MultigridBase<PressureType, PressureBCgpu, DIM3>::_dh;
    using TV = typename MultigridBase<PressureType, PressureBCgpu, DIM3>::TV;
    using T  = typename MultigridBase<PressureType, PressureBCgpu, DIM3>::T;

  public:
    double _dt;
    double _t;

  public:
    PressureSolverGPU(int3 dim, double3 dh, double dt)
        : MultigridBase<PressureType, PressureBCgpu, DIM3>(dim, dh), _dt(dt) {}

    virtual void smooth(PressureType& x, const PressureType& b, int3 dim, double3 dh, int n) override {
        for (int nn = 0; nn < n; nn++) {
            gpu::navier_stokes_flow::pressure::smooth<T, TV>(x.data(), b.data(), dim, dh);
        }
    }

    virtual void compute_residuals(const PressureType& x, const PressureType& b, PressureType& r, int3 dim,
                                   double3 dh) override {
        gpu::navier_stokes_flow::pressure::compute_residuals<T, TV>(x.data(), b.data(), r.data(), dim, dh);
    }

    void compute_b(PressureType& b, const VelocityType& u) {
        using U = std::decay_t<decltype(u.data()[0])>;
        gpu::navier_stokes_flow::pressure::compute_b<T, U>(b.data(), u.data(), _dim, _dh, _dt);
    }
};

} // namespace pangu

#endif
