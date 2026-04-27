/**
 * @file PoissonProblem3Dgpu.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-18
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */
#include <PhysicsSolver/multigrid/set_boundary_types.h>

#include "MultigridBase.h"

namespace pangu {
using BCVectorTypegpu = GpuVector<char, char>;

template <typename VectorType>
class PoissonProblem3Dgpu : public MultigridBase<VectorType, BCVectorTypegpu, DIM3> {
    using MultigridBase<VectorType, BCVectorTypegpu, DIM3>::_dim;
    using MultigridBase<VectorType, BCVectorTypegpu, DIM3>::_dh;
    using TV = typename MultigridBase<VectorType, BCVectorTypegpu, DIM3>::TV;
    using T  = typename MultigridBase<VectorType, BCVectorTypegpu, DIM3>::T;

  public:
    PoissonProblem3Dgpu(int3 dim, double3 dh) : MultigridBase<VectorType, BCVectorTypegpu, DIM3>(dim, dh) {}

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorTypegpu& bc, int3 dim, double3 dh,
                        int n) override {
        for (int nn = 0; nn < n; nn++) {
            gpu::poisson::smooth<T, TV, char>(x.data(), b.data(), bc.data(), dim, dh);
        }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorTypegpu& bc,
                                   int3 dim, double3 dh) override {
        gpu::poisson::compute_residuals<T, TV, char>(x.data(), b.data(), r.data(), bc.data(), dim, dh);
    }
};

} // namespace pangu