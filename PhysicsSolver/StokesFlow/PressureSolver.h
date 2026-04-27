/**
 * @file PressureSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-26
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PRESSURE_SOLVER_H__
#define __PRESSURE_SOLVER_H__

#include <PhysicsSolver/multigrid/MultigridBase.h>

namespace pangu {

template <typename PressureType, typename VelocityType, typename PressureBoundaryType>
class PressureSolver : public MultigridBase<PressureType, PressureBoundaryType, DIM3> {
    using MultigridBase<PressureType, PressureBoundaryType, DIM3>::_dim;
    using MultigridBase<PressureType, PressureBoundaryType, DIM3>::_dh;

  public:
    double _dt;
    double _t;

  public:
    PressureSolver(int3 dim, double3 dh, double dt)
        : MultigridBase<PressureType, PressureBoundaryType, DIM3>(dim, dh), _dt(dt) {}

    virtual void smooth(PressureType& x, const PressureType& b, const PressureBoundaryType& bc, int3 dim, double3 dh,
                        int n) override {
        auto _x  = x.data();
        auto _b  = b.data();
        auto _bc = bc.data();

        auto xstride = 1;
        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        for (int nn = 0; nn < n; nn++) {
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        auto centerindex = i * xstride + j * ystride + k * zstride;
                        if (centerindex % 2 == 0 || _bc[centerindex] == DIRICHLET) continue;

                        auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                        auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                        auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                        auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                        auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                        auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                        auto dx2 = dh.x * dh.x;
                        auto dy2 = dh.y * dh.y;
                        auto dz2 = dh.z * dh.z;

                        _x[i + j * dim.x + k * dim.x * dim.y]
                            = ((l + r) * dy2 * dz2 + (u + d) * dx2 * dz2 + (f + b) * dx2 * dy2
                               - _b[i + j * dim.x + k * dim.x * dim.y] * dz2 * dx2 * dy2)
                              / (2.0 * (dx2 * dy2 + dy2 * dz2 + dx2 * dz2));
                    }
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        auto centerindex = i + j * ystride + k * zstride;
                        if (centerindex % 2 == 1 || _bc[centerindex] == DIRICHLET) continue;

                        auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                        auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                        auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                        auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                        auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                        auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                        auto dx2 = dh.x * dh.x;
                        auto dy2 = dh.y * dh.y;
                        auto dz2 = dh.z * dh.z;

                        _x[i + j * dim.x + k * dim.x * dim.y]
                            = ((l + r) * dy2 * dz2 + (u + d) * dx2 * dz2 + (f + b) * dx2 * dy2
                               - _b[i + j * dim.x + k * dim.x * dim.y] * dz2 * dx2 * dy2)
                              / (2.0 * (dx2 * dy2 + dy2 * dz2 + dx2 * dz2));
                    }
        }
    }

    virtual void compute_residuals(const PressureType& x, const PressureType& b, PressureType& r,
                                   const PressureBoundaryType& bc, int3 dim, double3 dh) override {
        auto _x  = x.data();
        auto _b  = b.data();
        auto _r  = r.data();
        auto _bc = bc.data();

        auto dx2 = dh.x * dh.x;
        auto dy2 = dh.y * dh.y;
        auto dz2 = dh.z * dh.z;

        auto xstride = 1;
        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        for (int i = 0; i < dim.x; i++)
            for (int j = 0; j < dim.y; j++)
                for (int k = 0; k < dim.z; k++) {
                    auto centerindex = i * xstride + j * ystride + k * zstride;
                    if (_bc[centerindex] == DIRICHLET) continue;

                    auto m = _x[centerindex];
                    auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                    auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                    auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                    auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                    auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                    auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                    _r[centerindex]
                        = _b[centerindex] - (l + r - 2.0 * m) / dx2 - (u + d - 2.0 * m) / dy2 - (b + f - 2.0 * m) / dz2;
                }
    }

    void compute_b(PressureType& b, const VelocityType& u) {
        auto _b = b.data();
        auto _u = u.data();

        auto xstride = 1;
        auto ystride = _dim.x;
        auto zstride = _dim.x * _dim.y;

        for (int i = 0; i <= _dim.x - 2; i++) {
            for (int j = 0; j <= _dim.y - 2; j++) {
                for (int k = 0; k <= _dim.z - 2; k++) {
                    auto centerindex = i * xstride + j * ystride + k * zstride;

                    _b[centerindex] = ((_u[centerindex + 1].x - _u[centerindex].x) / _dh.x
                                       + (_u[centerindex + ystride].y - _u[centerindex].y) / _dh.y
                                       + (_u[centerindex + zstride].z - _u[centerindex].z) / _dh.z)
                                      / _dt;
                }
            }
        }
    }
};

} // namespace pangu

#endif
