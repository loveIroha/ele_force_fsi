/**
 * @file PressureSolverGPU.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-19
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PRESSURE_SOLVER_GPU_H__
#define __PRESSURE_SOLVER_GPU_H__

#include <GPU/gpu_lib.h>
#include <PhysicsSolver/multigrid/MultigridBase.h>

namespace pangu {

template <typename PressureType, typename VelocityType, typename PressureBoundaryType>
class PressureSolverGPU : public MultigridBase<PressureType, PressureBoundaryType, DIM3> {
    using MultigridBase<PressureType, PressureBoundaryType, DIM3>::_dim;
    using MultigridBase<PressureType, PressureBoundaryType, DIM3>::_dh;
    using TV = typename MultigridBase<PressureType, PressureBoundaryType, DIM3>::TV;
    using T  = typename MultigridBase<PressureType, PressureBoundaryType, DIM3>::T;

  public:
    double _dt;
    double _t;

  public:
    PressureSolverGPU(int3 dim, double3 dh, double dt)
        : MultigridBase<PressureType, PressureBoundaryType, DIM3>(dim, dh), _dt(dt) {}

    virtual void smooth(PressureType& x, const PressureType& b, const PressureBoundaryType& bc, int3 dim, double3 dh,
                        int n) override {
        for (int nn = 0; nn < n; nn++) {
            gpu::stokesflow::pressure::smooth<T, TV, char>(x.data(), b.data(), bc.data(), dim, dh);
        }
    }

    virtual void compute_residuals(const PressureType& x, const PressureType& b, PressureType& r,
                                   const PressureBoundaryType& bc, int3 dim, double3 dh) override {
        gpu::stokesflow::pressure::compute_residuals<T, TV, char>(x.data(), b.data(), r.data(), bc.data(), dim, dh);
    }

    void compute_b(PressureType& b, const VelocityType& u) {
        using U = std::decay_t<decltype(u.data()[0])>;
        gpu::stokesflow::pressure::compute_b<T, U>(b.data(), u.data(), _dim, _dh, _dt);
    }

    void compute_b_with_source(PressureType& b, const VelocityType& u, const PressureType& s) {
        using U = std::decay_t<decltype(u.data()[0])>;
        gpu::stokesflow::pressure::compute_b_with_source<T, U>(b.data(), u.data(), s.data(), _dim, _dh, _dt);
    }
};

} // namespace pangu

#endif
