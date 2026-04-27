/**
 * @file TentitiveVelocityGPU.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-14
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

// $$
// \begin{aligned}
// \frac{\partial \boldsymbol u}{\partial t}+\boldsymbol u\cdot\nabla\boldsymbol
// u&=\mu\Delta \boldsymbol u - \nabla p + f \nabla\cdot \boldsymbol u &= 0
// \end{aligned}
// $$

// $$
// \frac{\partial \boldsymbol u}{\partial t} = \mu\Delta \boldsymbol u + f
// $$

#ifndef __TENTITIVE_VELOCITY_GPU_H__
#define __TENTITIVE_VELOCITY_GPU_H__

#include <PhysicsSolver/multigrid/MultigridBase.h>

#include <memory>

namespace pangu {

using VelocityBCgpu = GpuVector<char, char3>;

template <typename PressureType, typename VelocityType>
class TentitiveVelocityGPU : public MultigridBase<VelocityType, VelocityBCgpu, DIM3> {
  private:
    using MultigridBase<VelocityType, VelocityBCgpu, DIM3>::_dim;
    using MultigridBase<VelocityType, VelocityBCgpu, DIM3>::_dh;

    using TV = typename MultigridBase<VelocityType, VelocityBCgpu, DIM3>::TV;
    using T  = typename MultigridBase<VelocityType, VelocityBCgpu, DIM3>::T;

  public:
    double _dt; // time step are needed, because smooth need it
    double _t;  // current time
    double _nu;

  public:
    TentitiveVelocityGPU(int3 dim, double3 dh, double dt, double nu)
        : MultigridBase<VelocityType, VelocityBCgpu, DIM3>(dim, dh), _dt(dt), _nu(nu) {}

    ~TentitiveVelocityGPU() {}

    virtual void smooth(VelocityType& x, const VelocityType& b, int3 dim, double3 dh, int n) {
        for (int nn = 0; nn < n; nn++) {
            gpu::navier_stokes_flow::velocity::smooth<T, TV>(x.data(), b.data(), dim, dh, _nu, _dt);
        }
    }

    virtual void compute_residuals(const VelocityType& x, const VelocityType& b, VelocityType& r, int3 dim,
                                   double3 dh) {
        gpu::navier_stokes_flow::velocity::compute_residuals<T, TV>(x.data(), b.data(), r.data(), dim, dh, _nu, _dt);
    }

    void compute_b(VelocityType& b, const VelocityType& f, const VelocityType& un, double t) {
        gpu::navier_stokes_flow::velocity::compute_b<T, TV>(b.data(), f.data(), un.data(), _dim, _dh, _dt);
    }

    void apply_dirichlet_bcs(VelocityType& u) { CHECK_F(false, "apply_dirichlet_bcs is not implemented!"); }
};

} // namespace pangu

#endif
