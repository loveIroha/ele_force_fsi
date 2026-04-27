/**
 * @file HeatEquation1D.h
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
class HeatEquation1D : public MultigridBase<VectorType, BCVectorType, DIM1> {
  private:
    using MultigridBase<VectorType, BCVectorType, DIM1>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM1>::_dh;

  public:
    double _dt; // time step
    double _t;  // current time

    HeatEquation1D(int3 dim, double3 dh, double dt) : MultigridBase<VectorType, BCVectorType, DIM1>(dim, dh), _dt(dt) {}

    ~HeatEquation1D() {}

    virtual void smooth(VectorType& x, const VectorType& b, int3 dim, double3 dh, int n) {
        CHECK_F(false, "Boundary conditions are not given.");
    }
    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bcs, int3 dim, double3 dh, int n) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _bcs = bcs.data();
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < dim.x; i++) {
                if (_bcs[i] == DIRICHLET) continue;
                auto l = i == 0 ? _x[i + 1] : _x[i - 1];
                auto r = i == dim.x - 1 ? _x[i - 1] : _x[i + 1];
                _x[i]  = (_b[i] + (l + r) / dh.x / dh.x) / (1.0 / _dt + 2.0 / dh.x / dh.x);
            }
        }
    }
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bcs,
                                   int3 dim, double3 dh) {
        auto _x   = x.data();
        auto _b   = b.data();
        auto _r   = r.data();
        auto _bcs = bcs.data();

        for (int i = 0; i < dim.x; i++) {
            if (_bcs[i] == DIRICHLET) continue;
            auto l = i == 0 ? _x[i + 1] : _x[i - 1];
            auto r = i == dim.x - 1 ? _x[i - 1] : _x[i + 1];
            auto m = _x[i];

            _r[i] = _b[i] - m / _dt + (l + r - 2.0 * m) / dh.x / dh.x;
        }
    }
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, int3 dim, double3 dh) {
        CHECK_F(false, "Boundary conditions are not given.");
    }

    void compute_b(VectorType& b, const VectorType& un, double t) {
        auto _b  = b.data();
        auto _un = un.data();
        for (int i = 0; i < _dim.x; i++) {
            double x = i * _dh.x;
            _b[i].x  = x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0 - t * (1.0 - 2.0 * x) + _un[i].x / _dt;
            _b[i].y  = x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0 - t * (1.0 - 2.0 * x) + _un[i].y / _dt;
            _b[i].z  = x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0 - t * (1.0 - 2.0 * x) + _un[i].z / _dt;
        }
    }

    void compute_exact(VectorType& ue, double t) {
        auto _ue = ue.data();
        for (int i = 0; i < _dim.x; i++) {
            double x = i * _dh.x;
            _ue[i].x = (x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0) * t;
            _ue[i].y = (x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0) * t;
            _ue[i].z = (x * x / 2.0 - x * x * x / 3.0 - 1.0 / 12.0) * t;
        }
    }
};

} // namespace pangu
