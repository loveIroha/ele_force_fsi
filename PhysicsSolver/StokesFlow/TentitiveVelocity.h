/**
 * @file TentitiveVelocity.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-25
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

#ifndef __TENTITIVE_VELOCITY_H__
#define __TENTITIVE_VELOCITY_H__

#include <PhysicsSolver/multigrid/MultigridBase.h>

#include <memory>

namespace pangu {

template <typename PressureType, typename VelocityType, typename VelocityBoundaryType>
class TentitiveVelocity : public MultigridBase<VelocityType, VelocityBoundaryType, DIM3> {
  private:
    using MultigridBase<VelocityType, VelocityBoundaryType, DIM3>::_dim;
    using MultigridBase<VelocityType, VelocityBoundaryType, DIM3>::_dh;

    using TV = typename MultigridBase<VelocityType, VelocityBoundaryType, DIM3>::TV;
    using T  = typename MultigridBase<VelocityType, VelocityBoundaryType, DIM3>::T;

  public:
    double _dt; // time step are needed, because smooth need it
    double _t;  // current time
    double _nu;

  public:
    TentitiveVelocity(int3 dim, double3 dh, double dt, double nu)
        : MultigridBase<VelocityType, VelocityBoundaryType, DIM3>(dim, dh), _dt(dt), _nu(nu) {}

    ~TentitiveVelocity() {}

    /// 还要判断边界条件。
    /// 边界条件如果是char3，那么速度的三个分量需要分开计算。
    ///        如果是char，那么速度的三个分量可以放一起计算。
    virtual void smooth(VelocityType& x, const VelocityType& b, int3 dim, double3 dh, int n) {
        CHECK_F(false, "Boundary conditions are not given.");
    }

    virtual void smooth(VelocityType& x, const VelocityType& b, const VelocityBoundaryType& bc, int3 dim, double3 dh,
                        int n) {
        auto _x  = x.data();
        auto _b  = b.data();
        auto _bc = bc.data();

        auto xstride = 1;
        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        int color = 0;

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        auto centerindex = i * xstride + j * ystride + k * zstride;
                        if ((centerindex + color) % 2 == 0 || _bc[centerindex] == DIRICHLET) continue;

                        auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                        auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                        auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                        auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                        auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                        auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                        _x[centerindex] = (_b[centerindex] + (l + r) * _nu / dh.x / dh.x + (u + d) * _nu / dh.y / dh.y
                                           + (f + b) * _nu / dh.z / dh.z)
                                          / (1.0 / _dt + _nu * 2.0 / dh.x / dh.x + _nu * 2.0 / dh.y / dh.y
                                             + _nu * 2.0 / dh.z / dh.z);
                    }
            color = 1;
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        auto centerindex = i * xstride + j * ystride + k * zstride;
                        if ((centerindex + color) % 2 == 0 || _bc[centerindex] == DIRICHLET) continue;

                        auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                        auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                        auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                        auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                        auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                        auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                        _x[centerindex] = (_b[centerindex] + (l + r) * _nu / dh.x / dh.x + (u + d) * _nu / dh.y / dh.y
                                           + (f + b) * _nu / dh.z / dh.z)
                                          / (1.0 / _dt + _nu * 2.0 / dh.x / dh.x + _nu * 2.0 / dh.y / dh.y
                                             + _nu * 2.0 / dh.z / dh.z);
                    }
        }
    }

    virtual void compute_residuals(const VelocityType& x, const VelocityType& b, VelocityType& r,
                                   const VelocityBoundaryType& bc, int3 dim, double3 dh) override {
        auto _x  = x.data();
        auto _b  = b.data();
        auto _r  = r.data();
        auto _bc = bc.data();

        auto xstride = 1;
        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        for (int i = 0; i < dim.x; i++)
            for (int j = 0; j < dim.y; j++)
                for (int k = 0; k < dim.z; k++) {
                    auto centerindex = i * xstride + j * dim.x + k * dim.x * dim.y;
                    if (_bc[centerindex] == DIRICHLET) continue; // Dirichlet boundary conditions.

                    auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                    auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                    auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                    auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                    auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                    auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                    _r[centerindex] = _b[centerindex] - _x[centerindex] / _dt
                                      + (l + r - 2.0 * _x[centerindex]) * _nu / dh.x / dh.x
                                      + (u + d - 2.0 * _x[centerindex]) * _nu / dh.y / dh.y
                                      + (f + b - 2.0 * _x[centerindex]) * _nu / dh.z / dh.z;
                }
    }

    void compute_b(VelocityType& b, const VelocityType& f, const VelocityType& un, double t) {
        auto _b  = b.data();
        auto _f  = f.data();
        auto _un = un.data();

        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    auto centerindex = i + j * _dim.x + k * _dim.x * _dim.y;
                    _b[centerindex]  = _f[centerindex] + _un[centerindex] / _dt;
                }
    }

    void apply_dirichlet_bcs(VelocityType& u) { CHECK_F(false, "To be implemented!"); }
};

} // namespace pangu

#endif
