/**
 * @file HeatEquation3D.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-03
 * @updated 2023-03-30 smooth 和 residual 中加入边界条件参数
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include "MultigridBase.h"

namespace pangu {

template <typename VectorType, typename BCVectorType>
class HeatEquation3D : public MultigridBase<VectorType, BCVectorType, DIM3> {
  private:
    using MultigridBase<VectorType, BCVectorType, DIM3>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM3>::_dh;
    using TV = typename MultigridBase<VectorType, BCVectorType, DIM3>::TV;
    using T  = typename MultigridBase<VectorType, BCVectorType, DIM3>::T;

  public:
    double _dt; // time step
    double _t;  // current time

    HeatEquation3D(int3 dim, double3 dh, double dt) : MultigridBase<VectorType, BCVectorType, DIM3>(dim, dh), _dt(dt) {}

    ~HeatEquation3D() {}

    virtual void smooth(VectorType& x, const VectorType& b, int3 dim, double3 dh, int n) {
        CHECK_F(false, "Boundary conditions are not given.");
    }
    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bcs, int3 dim, double3 dh, int n) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _bcs = bcs.data();

        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        for (int k = 0; k < n; k++)
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        if (_bcs[i + j * ystride + k * zstride] == DIRICHLET) continue;
                        auto centerindex = i + j * dim.x + k * dim.x * dim.y;

                        auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                        auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                        auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                        auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                        auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                        auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                        _x[centerindex]
                            = (_b[centerindex] + (l + r) / dh.x / dh.x + (u + d) / dh.y / dh.y + (f + b) / dh.z / dh.z)
                              / (1.0 / _dt + 2.0 / dh.x / dh.x + 2.0 / dh.y / dh.y + 2.0 / dh.z / dh.z);
                    }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bcs,
                                   int3 dim, double3 dh) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _r   = r.data();
        auto _bcs = bcs.data();

        auto ystride = dim.x;
        auto zstride = dim.x * dim.y;

        for (int i = 0; i < dim.x; i++)
            for (int j = 0; j < dim.y; j++)
                for (int k = 0; k < dim.z; k++) {
                    auto centerindex = i + j * dim.x + k * dim.x * dim.y;
                    if (_bcs[centerindex] == DIRICHLET) {
                        _r[centerindex] = TV();
                        continue;
                    }
                    auto l = i == 0 ? _x[centerindex + 1] : _x[centerindex - 1];
                    auto r = i == dim.x - 1 ? _x[centerindex - 1] : _x[centerindex + 1];
                    auto u = j == 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                    auto d = j == dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                    auto f = k == 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                    auto b = k == dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                    _r[centerindex] = _b[centerindex] - _x[centerindex] / _dt
                                      + (l + r - 2.0 * _x[centerindex]) / dh.x / dh.x
                                      + (u + d - 2.0 * _x[centerindex]) / dh.y / dh.y
                                      + (f + b - 2.0 * _x[centerindex]) / dh.z / dh.z;
                }
    }
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, int3 dim, double3 dh) {
        CHECK_F(false, "Boundary conditions are not given.");
    }

    void compute_b(VectorType& b, const VectorType& un, double t) {
        auto _b  = b.data();
        auto _un = un.data();

        auto ystride = _dim.x;
        auto zstride = _dim.x * _dim.y;

        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    double x = i * _dh.x;
                    double y = j * _dh.y;
                    double z = k * _dh.z;

                    auto centerindex = i + j * _dim.x + k * _dim.x * _dim.y;

                    _b[centerindex].x
                        = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                              * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * x) * (y * y / 2.0 - y * y * y / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * y) * (x * x / 2.0 - x * x * x / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * z) * (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                          + _un[centerindex].x / _dt;
                    _b[centerindex].y
                        = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                              * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * x) * (y * y / 2.0 - y * y * y / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * y) * (x * x / 2.0 - x * x * x / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * z) * (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                          + _un[centerindex].x / _dt;
                    _b[centerindex].z
                        = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                              * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * x) * (y * y / 2.0 - y * y * y / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * y) * (x * x / 2.0 - x * x * x / 3.0) * (z * z / 2.0 - z * z * z / 3.0)
                          - t * (1.0 - 2.0 * z) * (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                          + _un[centerindex].x / _dt;
                }
    }

    void compute_exact(VectorType& u_exact, double t) {
        auto _u_exact = u_exact.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    double x = i * _dh.x;
                    double y = j * _dh.y;
                    double z = k * _dh.z;

                    auto centerindex        = i + j * _dim.x + k * _dim.x * _dim.y;
                    _u_exact[centerindex].x = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                                              * (z * z / 2.0 - z * z * z / 3.0) * t;
                    _u_exact[centerindex].y = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                                              * (z * z / 2.0 - z * z * z / 3.0) * t;
                    _u_exact[centerindex].z = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                                              * (z * z / 2.0 - z * z * z / 3.0) * t;
                }
    }

    void apply_dirichlet_bcs(VectorType& u, const VectorType& u_exact, const BCVectorType& bcs) {
        auto _u       = u.data();
        auto _u_exact = u_exact.data();
        auto _bcs     = bcs.data();

        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    auto centerindex = i + j * _dim.x + k * _dim.x * _dim.y;
                    if (_bcs[centerindex] == DIRICHLET) _u[centerindex] = _u_exact[centerindex];
                }
    }
};
} // namespace pangu
