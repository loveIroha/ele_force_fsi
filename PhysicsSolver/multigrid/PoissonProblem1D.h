/**
 * @file PoissonProblem1D .h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-13
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */
#include "MultigridBase.h"

//  $$
//  \Delta u = f
//  $$

namespace pangu {
template <typename VectorType, typename BCVectorType>
class PoissonProblem1D : public MultigridBase<VectorType, BCVectorType, DIM1> {
    using MultigridBase<VectorType, BCVectorType, DIM1>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM1>::_dh;

  public:
    PoissonProblem1D(int3 dim, double3 dh) : MultigridBase<VectorType, BCVectorType, DIM1>(dim, dh) {}

    virtual void smooth(VectorType& x, const VectorType& b, int3 dim, double3 dh, int n) {
        auto _x = flatten(x).data;
        auto _b = flatten(b).data;
        for (int k = 0; k < n; k++)
            for (int i = 1; i < dim.x - 1; i++) {
                _x[i] = 0.5 * (_x[i - 1] + _x[i + 1] - dh.x * dh.x * _b[i]);
            }
    }

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh, int n) {
        auto _x  = flatten(x).data;
        auto _b  = flatten(b).data;
        auto _bc = flatten(bc).data;
        for (int k = 0; k < n; k++)
            for (int i = 0; i < dim.x; i++) {
                if (_bc[i] == DIRICHLET) continue;
                auto l = i - 1 < 0 ? _x[i + 1] : _x[i - 1];
                auto r = i + 1 > dim.x - 1 ? _x[i - 1] : _x[i + 1];
                _x[i]  = 0.5 * (l + r - dh.x * dh.x * _b[i]);
            }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, int3 dim, double3 dh) {
        auto _x = flatten(x).data;
        auto _b = flatten(b).data;
        auto _r = flatten(r).data;

        for (int i = 1; i < dim.x - 1; i++) {
            _r[i] = _b[i] - 1.0 / dh.x / dh.x * (_x[i - 1] + _x[i + 1] - 2.0 * _x[i]);
        }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh) {
        auto _x  = flatten(x).data;
        auto _b  = flatten(b).data;
        auto _r  = flatten(r).data;
        auto _bc = flatten(bc).data;

        for (int i = 0; i < dim.x; i++) {
            if (_bc[i] == DIRICHLET) continue;

            auto l = i - 1 < 0 ? _x[i + 1] : _x[i - 1];
            auto r = i + 1 > dim.x - 1 ? _x[i - 1] : _x[i + 1];

            _r[i] = _b[i] - 1.0 / dh.x / dh.x * (l + r - 2.0 * _x[i]);
        }
    }

    void compute_b(VectorType& b) {
        auto   _b = flatten(b).data;
        double pi = 3.14159265358979;
        double x;
        for (int i = 0; i < _dim.x; i++) {
            x = i * _dh.x;
            // _b[i] = -pi*pi*std::sin(pi*x);
            _b[i] = -pi * pi * std::cos(pi * x);
            // _b[i] = (x*x+x-1.0)*std::exp(x);
            // _b[i] = std::exp(x);
        }
    }

    void compute_exact(VectorType& b) {
        auto   _b = flatten(b).data;
        double pi = 3.14159265358979;
        double x;
        for (int i = 0; i < _dim.x; i++) {
            x = i * _dh.x;
            // _b[i] = std::sin(pi*x);
            _b[i] = std::cos(pi * x);
            // _b[i] = std::exp(x)*(x*x-x);
            // _b[i] = std::exp(x);
        }
    }

    void apply_dirichlet_bcs(VectorType& x, const VectorType& bc_values) {
        auto _bc       = flatten(bc_values).data;
        auto _x        = flatten(x).data;
        _x[0]          = _bc[0];
        _x[_dim.x - 1] = _bc[_dim.x - 1];
    }
};

} // namespace pangu