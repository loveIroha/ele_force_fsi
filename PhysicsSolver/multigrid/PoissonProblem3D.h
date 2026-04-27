/**
 * @file PoissonProblem3D.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-13
 * @updated 2023-03-29 achieve ideal convergence rate
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */
#include "MultigridBase.h"

namespace pangu {
template <typename VectorType, typename BCVectorType>
class PoissonProblem3D : public MultigridBase<VectorType, BCVectorType, DIM3> {
    using MultigridBase<VectorType, BCVectorType, DIM3>::_dim;
    using MultigridBase<VectorType, BCVectorType, DIM3>::_dh;
    double pi = 3.14159265358979;

  public:
    PoissonProblem3D(int3 dim, double3 dh) : MultigridBase<VectorType, BCVectorType, DIM3>(dim, dh) {}

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh,
                        int n) override {
        auto _x  = x.data();
        auto _b  = b.data();
        auto _bc = bc.data();

        for (int nn = 0; nn < n; nn++) {
            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        if ((i + j * dim.x + k * dim.x * dim.y) % 2 == 0) {
                            if (_bc[i + j * dim.x + k * dim.x * dim.y] == DIRICHLET) continue;

                            auto l = i - 1 < 0 ? _x[i + 1 + j * dim.x + k * dim.x * dim.y]
                                               : _x[i - 1 + j * dim.x + k * dim.x * dim.y];
                            auto r = i + 1 > dim.x - 1 ? _x[i - 1 + j * dim.x + k * dim.x * dim.y]
                                                       : _x[i + 1 + j * dim.x + k * dim.x * dim.y];
                            auto d = j - 1 < 0 ? _x[i + (j + 1) * dim.x + k * dim.x * dim.y]
                                               : _x[i + (j - 1) * dim.x + k * dim.x * dim.y];
                            auto u = j + 1 > dim.y - 1 ? _x[i + (j - 1) * dim.x + k * dim.x * dim.y]
                                                       : _x[i + (j + 1) * dim.x + k * dim.x * dim.y];
                            auto b = k - 1 < 0 ? _x[i + j * dim.x + (k + 1) * dim.x * dim.y]
                                               : _x[i + j * dim.x + (k - 1) * dim.x * dim.y];
                            auto f = k + 1 > dim.z - 1 ? _x[i + j * dim.x + (k - 1) * dim.x * dim.y]
                                                       : _x[i + j * dim.x + (k + 1) * dim.x * dim.y];

                            auto dx2 = dh.x * dh.x;
                            auto dy2 = dh.y * dh.y;
                            auto dz2 = dh.z * dh.z;

                            _x[i + j * dim.x + k * dim.x * dim.y]
                                = ((l + r) * dy2 * dz2 + (u + d) * dx2 * dz2 + (f + b) * dx2 * dy2
                                   - _b[i + j * dim.x + k * dim.x * dim.y] * dz2 * dx2 * dy2)
                                  / (2.0 * (dx2 * dy2 + dy2 * dz2 + dx2 * dz2));
                        }
                    }

            for (int i = 0; i < dim.x; i++)
                for (int j = 0; j < dim.y; j++)
                    for (int k = 0; k < dim.z; k++) {
                        if ((i + j * dim.x + k * dim.x * dim.y) % 2 == 1) {
                            if (_bc[i + j * dim.x + k * dim.x * dim.y] == DIRICHLET) continue;

                            auto l = i - 1 < 0 ? _x[i + 1 + j * dim.x + k * dim.x * dim.y]
                                               : _x[i - 1 + j * dim.x + k * dim.x * dim.y];
                            auto r = i + 1 > dim.x - 1 ? _x[i - 1 + j * dim.x + k * dim.x * dim.y]
                                                       : _x[i + 1 + j * dim.x + k * dim.x * dim.y];
                            auto d = j - 1 < 0 ? _x[i + (j + 1) * dim.x + k * dim.x * dim.y]
                                               : _x[i + (j - 1) * dim.x + k * dim.x * dim.y];
                            auto u = j + 1 > dim.y - 1 ? _x[i + (j - 1) * dim.x + k * dim.x * dim.y]
                                                       : _x[i + (j + 1) * dim.x + k * dim.x * dim.y];
                            auto b = k - 1 < 0 ? _x[i + j * dim.x + (k + 1) * dim.x * dim.y]
                                               : _x[i + j * dim.x + (k - 1) * dim.x * dim.y];
                            auto f = k + 1 > dim.z - 1 ? _x[i + j * dim.x + (k - 1) * dim.x * dim.y]
                                                       : _x[i + j * dim.x + (k + 1) * dim.x * dim.y];

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
    }

    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh) {
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
                    auto l = i - 1 < 0 ? _x[centerindex + xstride] : _x[centerindex - xstride];
                    auto r = i + 1 > dim.x - 1 ? _x[centerindex - xstride] : _x[centerindex + xstride];
                    auto d = j - 1 < 0 ? _x[centerindex + ystride] : _x[centerindex - ystride];
                    auto u = j + 1 > dim.y - 1 ? _x[centerindex - ystride] : _x[centerindex + ystride];
                    auto b = k - 1 < 0 ? _x[centerindex + zstride] : _x[centerindex - zstride];
                    auto f = k + 1 > dim.z - 1 ? _x[centerindex - zstride] : _x[centerindex + zstride];

                    _r[centerindex]
                        = _b[centerindex] - (l + r - 2.0 * m) / dx2 - (u + d - 2.0 * m) / dy2 - (b + f - 2.0 * m) / dz2;
                }
    }

    void compute_b(VectorType& b) {
        auto _b = b.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    double x = i * _dh.x;
                    double y = j * _dh.y;
                    double z = k * _dh.z;
                    // _b[i+j*_dim.x+k*_dim.x*_dim.y] = z*z*y*y*std::exp(x*y*z) +
                    // x*x*y*y*std::exp(x*y*z) + z*z*x*x*std::exp(x*y*z);
                    // _b[i+j*_dim.x+k*_dim.x*_dim.y] = z*z*y*y*std::exp(x*y*z) +
                    // x*x*y*y*std::exp(x*y*z) + z*z*x*x*std::exp(x*y*z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].x
                        = -3.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].y
                        = -3.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].z
                        = -3.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                }
    }

    void compute_exact(VectorType& b) {
        auto _b = b.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    double x = i * _dh.x;
                    double y = j * _dh.y;
                    double z = k * _dh.z;
                    // _b[i+j*_dim.x+k*_dim.x*_dim.y] = std::exp(x*y*z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].x = std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].y = std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                    _b[i + j * _dim.x + k * _dim.x * _dim.y].z = std::cos(pi * x) * std::cos(pi * y) * std::cos(pi * z);
                }
    }

    void apply_dirichlet_bcs(VectorType& x, const VectorType& bc_values) {
        auto _bc = bc_values.data();
        auto _x  = x.data();
        for (int i = 0; i < _dim.x; i++)
            for (int j = 0; j < _dim.y; j++)
                for (int k = 0; k < _dim.z; k++) {
                    _x[i + j * _dim.x + k * _dim.x * _dim.z] = _bc[i + j * _dim.x + k * _dim.x * _dim.y];
                }

        for (int i = 1; i < _dim.x - 1; i++)
            for (int j = 1; j < _dim.y - 1; j++)
                for (int k = 1; k < _dim.z - 1; k++) {
                    _x[i + j * _dim.x + k * _dim.x * _dim.y] = make_double3(0, 0, 0);
                }
    }
};

} // namespace pangu