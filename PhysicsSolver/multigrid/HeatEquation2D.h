/**
 * @file HeatEquation2D.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-25
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

// 1. neuman boundary condition at left side.
// 2. solve time dependent PDE equations.
// 3. use TV type unkowns.

// $$
// \frac{\partial u}{\partial t}=\Delta u + f
// $$
// $$
// \frac{u_i^{n+1}-u_i^n}{\Delta
// t}=\frac{u^{n+1}_{i-1}-2u^{n+1}_i+u^{n+1}_{i+1}}{\Delta x^2}+f_i
// $$
// $$
// \frac{u_i^{n+1}}{\Delta
// t}-\frac{u^{n+1}_{i-1}-2u^{n+1}_i+u^{n+1}_{i+1}}{\Delta
// x^2}=f_i+\frac{u_i^n}{\Delta t}
// $$
// $$
// \frac{u_i}{\Delta t}-\frac{u_{i-1}-2u_i+u_{i+1}}{\Delta x^2}=b_i
// $$
// $$
// (\frac{1}{\Delta t}+\frac{2}{\Delta
// x^2})u_i=b_i+\frac{u_{i-1}+u_{i+1}}{\Delta x^2}
// $$
// 高斯塞德尔迭代
// $$
// u_i=\left(b_i+\frac{u_{i-1}+u_{i+1}}{\Delta x^2}\right)/\left(\frac{1}{\Delta
// t}+\frac{2}{\Delta x^2}\right)
// $$
// 残差的计算
// $$
// r_i=b_i-\frac{u_i}{\Delta t}+\frac{u_{i-1}-2u_i+u_{i+1}}{\Delta x^2}
// $$

#include "MultigridBase.h"

namespace pangu {

template <typename VectorType, typename BCVectorType>
class HeatEquation2D : public MultigridBase<VectorType, BCVectorType, DIM2> {
  private:
    using MultigridBase<VectorType, BCVectorType, DIM2>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM2>::_dh;

  public:
    double _dt; // time step
    double _t;  // current time

    HeatEquation2D(int3 dim, double3 dh, double dt) : MultigridBase<VectorType, BCVectorType, DIM2>(dim, dh), _dt(dt) {}

    ~HeatEquation2D() {}

    virtual void smooth(VectorType& x, const VectorType& b, int3 dim, double3 dh, int n) {
        CHECK_F(false, "Boundary conditions are not given.");
    }
    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bcs, int3 dim, double3 dh, int n) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _bcs = bcs.data();
        for (int k = 0; k < n; k++)
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++) {
                    if (_bcs[i + j * dim.x] == DIRICHLET) continue;
                    auto l            = i == 0 ? _x[i + 1 + j * dim.x] : _x[i - 1 + j * dim.x];
                    auto r            = i == dim.x - 1 ? _x[i - 1 + j * dim.x] : _x[i + 1 + j * dim.x];
                    auto u            = j == 0 ? _x[i + (j + 1) * dim.x] : _x[i + (j - 1) * dim.x];
                    auto d            = j == dim.y - 1 ? _x[i + (j - 1) * dim.x] : _x[i + (j + 1) * dim.x];
                    _x[i + j * dim.x] = (_b[i + j * dim.x] + (l + r) / dh.x / dh.x + (u + d) / dh.y / dh.y)
                                        / (1.0 / _dt + 2.0 / dh.x / dh.x + 2.0 / dh.y / dh.y);
                }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bcs,
                                   int3 dim, double3 dh) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _r   = r.data();
        auto _bcs = bcs.data();

        for (int i = 0; i < dim.x; i++)
            for (int j = 0; j < dim.y; j++) {
                if (_bcs[i + j * dim.x] == DIRICHLET) continue;
                auto l = i == 0 ? _x[i + 1 + j * dim.x] : _x[i - 1 + j * dim.x];
                auto r = i == dim.x - 1 ? _x[i - 1 + j * dim.x] : _x[i + 1 + j * dim.x];
                auto u = j == 0 ? _x[i + (j + 1) * dim.x] : _x[i + (j - 1) * dim.x];
                auto d = j == dim.y - 1 ? _x[i + (j - 1) * dim.x] : _x[i + (j + 1) * dim.x];
                auto m = _x[i + j * dim.x];

                _r[i + j * dim.x] = _b[i + j * dim.x] - _x[i + j * dim.x] / _dt + (l + r - 2.0 * m) / dh.x / dh.x
                                    + (u + d - 2.0 * m) / dh.y / dh.y;
            }
    }
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, int3 dim, double3 dh) {
        CHECK_F(false, "Boundary conditions are not given.");
    }

    void compute_b(VectorType& b, const VectorType& un, double t) {
        auto _b  = b.data();
        auto _un = un.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                double x = i * _dh.x;
                double y = j * _dh.y;

                _b[i + j * _dim.x].x = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0)
                                       - t * (1.0 - 2.0 * x) * (y * y / 2.0 - y * y * y / 3.0)
                                       - t * (1.0 - 2.0 * y) * (x * x / 2.0 - x * x * x / 3.0)
                                       + _un[i + j * _dim.x].x / _dt;
            }
    }

    void compute_exact(VectorType& u_exact, double t) {
        auto _u_exact = u_exact.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                double x = i * _dh.x;
                double y = j * _dh.y;

                _u_exact[i + j * _dim.x].x = (x * x / 2.0 - x * x * x / 3.0) * (y * y / 2.0 - y * y * y / 3.0) * t;
            }
    }

    void apply_dirichlet_bcs(VectorType& u, const VectorType& u_exact, const BCVectorType& bcs) {
        auto _u       = u.data();
        auto _u_exact = u_exact.data();
        auto _bcs     = bcs.data();

        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                if (_bcs[i + j * _dim.x] == DIRICHLET) _u[i + j * _dim.x].x = _u_exact[i + j * _dim.x].x;
            }
    }
};
} // namespace pangu
