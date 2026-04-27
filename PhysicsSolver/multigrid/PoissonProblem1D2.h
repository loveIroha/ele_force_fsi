/**
 * @file PoissonProblem1D2 .h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-13
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */
#include "MultigridBase2.h"

namespace pangu {
template <typename VectorType, typename BCVectorType>
class PoissonProblem1D2 : public MultigridBase2<VectorType, BCVectorType, DIM1> {
    using MultigridBase2<VectorType, BCVectorType, DIM1>::_dim;
    using MultigridBase2<VectorType, BCVectorType, DIM1>::_dh;

  public:
    PoissonProblem1D2(int3 dim, double3 dh) : MultigridBase2<VectorType, BCVectorType, DIM1>(dim, dh) {}

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh, int n) {
        auto                _x  = flatten(x).data;
        auto                _b  = flatten(b).data;
        auto                _bc = flatten(bc).data;
        std::vector<double> x_new(dim.x + 2);
        auto                _x_new = x_new.data();

        for (int k = 0; k < n; k++) {
            _x[0]          = -_x[1];
            _x[_dim.x + 1] = -_x[_dim.x];
            for (int i = 1; i < dim.x + 1; i++) {
                auto l    = _x[i - 1];
                auto r    = _x[i + 1];
                _x_new[i] = 0.5 * (l + r - dh.x * dh.x * _b[i]);
            }

            for (int i = 0; i < dim.x + 2; i++) {
                _x[i] = _x_new[i];
            }
        }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh) {
        auto _x        = flatten(x).data;
        auto _b        = flatten(b).data;
        auto _r        = flatten(r).data;
        auto _bc       = flatten(bc).data;
        _x[0]          = -_x[1];
        _x[_dim.x + 1] = -_x[_dim.x];
        for (int i = 1; i < dim.x + 1; i++) {
            auto l = _x[i - 1];
            auto r = _x[i + 1];
            _r[i]  = _b[i] - 1.0 / dh.x / dh.x * (l + r - 2.0 * _x[i]);
        }
    }

    void compute_b(VectorType& b) {
        auto   _b = flatten(b).data;
        double pi = 3.14159265358979;
        double x;
        for (int i = 1; i < _dim.x + 1; i++) {
            x = i * _dh.x - 0.5 * _dh.x;
            // _b[i] = - pi * pi * std::sin(pi * x);
            // _b[i] = -pi * pi * std::cos(pi * x);
            _b[i] = (x * x - x - 1) * std::exp(x);
            // _b[i] = std::exp(x);
        }
    }

    void compute_exact(VectorType& b) {
        auto   _b = flatten(b).data;
        double pi = 3.14159265358979;
        double x;
        for (int i = 1; i < _dim.x + 1; i++) {
            x = i * _dh.x - 0.5 * _dh.x;
            // _b[i] = std::sin(pi * x);
            // _b[i] = std::cos(pi * x);
            _b[i] = (x * x - x) * std::exp(x);
            // _b[i] = std::exp(x);
        }
    }
};

} // namespace pangu