/**
 * @file PoissonProblem2D.h
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
class PoissonProblem2D : public MultigridBase<VectorType, BCVectorType, DIM2> {
    using MultigridBase<VectorType, BCVectorType, DIM2>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM2>::_dh;
    double pi = 3.14159265358979;

  public:
    PoissonProblem2D(int3 dim, double3 dh) : MultigridBase<VectorType, BCVectorType, DIM2>(dim, dh) {}

    virtual void smooth(VectorType& x, const VectorType& b, int3 dim, double3 dh, int n) {}

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh, int n) {
        auto _x  = flatten(x).data;
        auto _b  = flatten(b).data;
        auto _bc = flatten(bc).data;

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++) {
                    if (_bc[i + j * dim.x] == DIRICHLET) continue;
                    if ((i + j * dim.x) % 2 == 1) {
                        auto l   = i - 1 < 0 ? _x[i + 1 + j * dim.x] : _x[i - 1 + j * dim.x];
                        auto r   = i + 1 > dim.x - 1 ? _x[i - 1 + j * dim.x] : _x[i + 1 + j * dim.x];
                        auto d   = j - 1 < 0 ? _x[i + (j + 1) * dim.x] : _x[i + (j - 1) * dim.x];
                        auto u   = j + 1 > dim.y - 1 ? _x[i + (j - 1) * dim.x] : _x[i + (j + 1) * dim.x];
                        auto dy2 = dh.y * dh.y;
                        auto dx2 = dh.x * dh.x;
                        _x[i + j * dim.x]
                            = ((l + r) * dy2 + (u + d) * dx2 - _b[i + j * dim.x] * dx2 * dy2) / (2.0 * (dx2 + dy2));
                    }
                }
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++) {
                    if (_bc[i + j * dim.x] == DIRICHLET) continue;
                    if ((i + j * dim.x) % 2 == 0) {
                        auto l   = i - 1 < 0 ? _x[i + 1 + j * dim.x] : _x[i - 1 + j * dim.x];
                        auto r   = i + 1 > dim.x - 1 ? _x[i - 1 + j * dim.x] : _x[i + 1 + j * dim.x];
                        auto d   = j - 1 < 0 ? _x[i + (j + 1) * dim.x] : _x[i + (j - 1) * dim.x];
                        auto u   = j + 1 > dim.y - 1 ? _x[i + (j - 1) * dim.x] : _x[i + (j + 1) * dim.x];
                        auto dy2 = dh.y * dh.y;
                        auto dx2 = dh.x * dh.x;
                        _x[i + j * dim.x]
                            = ((l + r) * dy2 + (u + d) * dx2 - _b[i + j * dim.x] * dx2 * dy2) / (2.0 * (dx2 + dy2));
                    }
                }
        }
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, int3 dim, double3 dh) {}

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh) {
        auto _x  = flatten(x).data;
        auto _b  = flatten(b).data;
        auto _r  = flatten(r).data;
        auto _bc = flatten(bc).data;

        for (int i = 0; i < dim.x; i++)
            for (int j = 0; j < dim.y; j++) {
                if (_bc[i + j * dim.x] == DIRICHLET) continue;

                auto l = i - 1 < 0 ? _x[i + 1 + j * dim.x] : _x[i - 1 + j * dim.x];
                auto r = i + 1 > dim.x - 1 ? _x[i - 1 + j * dim.x] : _x[i + 1 + j * dim.x];
                auto d = j - 1 < 0 ? _x[i + (j + 1) * dim.x] : _x[i + (j - 1) * dim.x];
                auto u = j + 1 > dim.y - 1 ? _x[i + (j - 1) * dim.x] : _x[i + (j + 1) * dim.x];

                auto m            = _x[i + j * dim.x];
                auto dy2          = dh.y * dh.y;
                auto dx2          = dh.x * dh.x;
                _r[i + j * dim.x] = _b[i + j * dim.x] - (l + r - 2.0 * m) / dx2 - (u + d - 2.0 * m) / dy2;
            }
    }

    void compute_b(VectorType& b) {
        auto _b = flatten(b).data;
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                double x = i * _dh.x;
                double y = j * _dh.y;
                // _b[i+j*_dim.x] = x*x*std::exp(x*y) + y*y*std::exp(x*y);
                _b[i + j * _dim.x] = -2.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y);
            }
    }

    void compute_exact(VectorType& b) {
        auto _b = flatten(b).data;
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                double x           = i * _dh.x;
                double y           = j * _dh.y;
                _b[i + j * _dim.x] = std::cos(pi * x) * std::cos(pi * y);
                // _b[i+j*_dim.x] = std::exp(x*y);
            }
    }

    void apply_dirichlet_bcs(VectorType& x, const VectorType& bc_values, const BCVectorType& bc_types) {
        auto _bc   = flatten(bc_values).data;
        auto _bc_t = flatten(bc_types).data;
        auto _x    = flatten(x).data;
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++) {
                if (_bc_t[i + j * _dim.x] == DIRICHLET) _x[i + j * _dim.x] = _bc[i + j * _dim.x];
            }
    }
};

} // namespace pangu