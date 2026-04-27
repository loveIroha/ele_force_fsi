#include <boost/multi_array.hpp>

#include <io/loguru.hpp>
#include <io/writeVTK.h>
#include <vector_functions.h>

#include <cmath>

// TODO : This file should be reviewed and refactored carafully!
namespace interactor {
template <int N, typename T>
void outer_product(const T* v1, T* v2, T* out) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            out[N * j + i] = v1[i] * v2[j];
        }
}

// Standard 4-point Kernel for IBM
double phi_ib4(double r) {
    r         = fabs(r);
    double r2 = r * r;
    double phi;

    if (r < 1) {
        phi = (3 - 2 * r + sqrt(1 + 4 * r - 4 * r2)) / 8.;
    } else if (r >= 1 && r < 2) {
        phi = (5 - 2 * r - sqrt(-7 + 12 * r - 4 * r2)) / 8.;
    } else {
        phi = 0.;
    }

    return phi;
}

template <int N, typename T>
void phi_ib4_boundary(T* v, T A) {
    T phi_3;

    if (A > 2 - std::sqrt(0.5)) {
        phi_3 = 0.25 * (A - 1) - 0.125 * std::sqrt(-14 + 16 * A - 4 * A * A);
    } else {
        phi_3 = 0.25 * (A - 1);
    }
    v[3] = phi_3;
    v[0] = 1 - 0.5 * A + phi_3;
    v[1] = 0.5 - phi_3;
    v[2] = phi_3;

    LOG_SCOPE_FUNCTION(INFO);
}

template <int N, typename T>
void phi_function_weights(T* v, const T* r, T x, T width) {
    // LOG_SCOPE_FUNCTION(INFO);
    // LOG_F(INFO, "x : %f, width : %f", x, width);
    //////////////////  TODO : delta function should be dealt carefully when the
    /// solid move to near the boundary  //////////////////////
    // if (x < 1.5)
    // {
    //     phi_ib4_boundary<N>(v, x);
    //     return;
    // }
    // if (width - x - 1 < 1.5)
    // {
    //     T tmp_v[N];
    //     phi_ib4_boundary<N>(tmp_v, width - x);
    //     v[0] = tmp_v[3];
    //     v[1] = tmp_v[2];
    //     v[2] = tmp_v[1];
    //     v[3] = tmp_v[0];
    //     return;
    // }

    for (size_t i = 0; i < N; i++) {
        v[i] = phi_ib4(r[i]);
    }
}

template <int N, typename T>
void delta_function_weights(T* out, const T& r1, const T& r2, const T& x, const T& y, const T& width, const T& height) {
    const double r1_v[N] = {-1 - r1, -r1, 1 - r1, 2 - r1};
    const double r2_v[N] = {-1 - r2, -r2, 1 - r2, 2 - r2};

    // printf("r1_v[0] : %f\n", r1_v[0]);
    // printf("r1_v[1] : %f\n", r1_v[1]);
    // printf("r1_v[2] : %f\n", r1_v[2]);
    // printf("r1_v[3] : %f\n", r1_v[3]);

    double v1[N] = {};
    double v2[N] = {};

    phi_function_weights<N>(v1, r1_v, x, width);
    phi_function_weights<N>(v2, r2_v, y, height);

    outer_product<N>(v1, v2, out);
}

} // namespace interactor

template <typename TV1, typename TV2>
double ibm_delta2(const TV1& x, const TV2& g) {
    return interactor::phi_ib4(abs(x.x - g.x)) * interactor::phi_ib4(abs(x.y - g.y));
}

/// @brief
///
/// @param lagrange_u   Lagrange 速度分量 u
/// @param eulerian_u   Eulerian 速度分量 u
/// @param num          Lagrange 点的数量
/// @param dim          Eulerian 网格大小
/// @param h            Eulerian 网格密度
/// @param quadrature_rules
///
void interpolate_u(double* lagrange_u, const double* eulerian_u, int num, int2 dim, double hx, double hy,
                   const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);

    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];

        // i,j 为 u 在数组中的索引，而不是所在单元格的索引
        // dim 为 u 的数量，而不是单元格的数量
        // cell[i,j] u[i,j+1], [i*hx,j*hy+1/2]
        double x  = pos.x / hx;
        double y  = (pos.y + 0.5 * hy) / hy;
        int    i  = floor(x);
        int    j  = floor(y);
        double r1 = x - i;
        double r2 = y - j;

        double w[16] = {};
        double u[16] = {};
        interactor::delta_function_weights<4>(w, r1, r2, x, y, (double)dim.x, (double)dim.y);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) {
                    printf("出边界了！\n");
                    continue;
                }
                int ridx                   = (ii + i) + (jj + j) * dim.x;
                u[4 * (jj + 1) + (ii + 1)] = eulerian_u[ridx];
            }
        }

        double sum{};
        for (size_t i = 0; i < 16; i++)
            sum += u[i] * w[i];

        lagrange_u[gidx] = sum;
    }
}

void interpolate_v(double* lagrange_v, const double* eulerian_v, int num, int2 dim, double hx, double hy,
                   const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];

        double x  = (pos.x + 0.5 * hx) / hx;
        double y  = pos.y / hy;
        int    i  = floor(x);
        int    j  = floor(y);
        double r1 = x - i;
        double r2 = y - j;

        double w[16] = {};
        double u[16] = {};
        interactor::delta_function_weights<4>(w, r1, r2, x, y, (double)dim.x, (double)dim.y);

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) {
                    printf("出边界了！\n");
                    continue;
                }

                int ridx                   = (ii + i) + (jj + j) * dim.x;
                u[4 * (jj + 1) + (ii + 1)] = eulerian_v[ridx];
                // int ridx = ii + jj * dim.x;
                // double weight = ibm_delta2(make_double2((pos.x + 0.5 * hx) /
                // hx, pos.y / hy), make_double2(ii, jj)); sum += weight *
                // eulerian_v[ridx];
            }
        }
        double sum{};
        for (size_t i = 0; i < 16; i++)
            sum += u[i] * w[i];

        lagrange_v[gidx] = sum;
    }
}

void distribute_u(const double* lagrange_u, double* eulerian_u, int num, int2 dim, double hx, double hy,
                  const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos = quadrature_rules[gidx];
        double  x   = pos.x / hx;
        double  y   = (pos.y + 0.5 * hy) / hy;
        int     i   = floor(x);
        int     j   = floor(y);
        double  r1  = x - i;
        double  r2  = y - j;

        // double w[16] = {};
        // double u[16] = {};

        // int i = floor(pos.x / hx);
        // int j = floor((pos.y + 0.5 * hy) / hy);

        // double r1 = pos.x / hx - i;
        // double r2 = (pos.y + 0.5 * hy) / hy - j;

        double w[16] = {};
        double f[16] = {};
        interactor::delta_function_weights<4>(w, r1, r2, x, y, (double)dim.x, (double)dim.y);

        double F      = lagrange_u[gidx];
        double inv_h2 = 1.0 / hx / hy;

        // printf("F : %f\n", F);
        // printf("inv_h2 : %f\n", inv_h2);
        // for (size_t i = 0; i < 16; i++)
        // {
        //     printf("w : %f ", w[i]);
        // }

        for (size_t i = 0; i < 16; i++) {
            f[i] = inv_h2 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) { continue; }
                if (jj >= dim.y) printf("jy出边界了！\n");
                int ridx = (ii + i) + (jj + j) * dim.x;
                eulerian_u[ridx] += f[4 * (jj + 1) + (ii + 1)];
            }
        }
    }
}

void distribute_v(const double* lagrange_v, double* eulerian_v, int num, int2 dim, double hx, double hy,
                  const double4* quadrature_rules) {
    LOG_SCOPE_FUNCTION(INFO);
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos    = quadrature_rules[gidx];
        int     i      = floor((pos.x + 0.5 * hx) / hx);
        int     j      = floor((pos.y) / hy);
        double  inv_h2 = 1.0 / hx / hy;

        double r1 = (pos.x + 0.5 * hx) / hx - i;
        double r2 = (pos.y) / hy - j;

        double w[16] = {};
        double f[16] = {};

        interactor::delta_function_weights<4>(w, r1, r2, (pos.x + 0.5 * hx) / hx, (pos.y) / hy, (double)dim.x,
                                              (double)dim.y);

        double F = lagrange_v[gidx];

        for (size_t i = 0; i < 16; i++) {
            f[i] = inv_h2 * F * w[i] * pos.w;
        }

        for (int jj = -1; jj <= 2; jj++) {
            for (int ii = -1; ii <= 2; ii++) {
                if (!((jj + j) >= 0 && (jj + j) < dim.y && (ii + i) >= 0 && (ii + i) < dim.x)) { continue; }
                if (jj >= dim.y) printf("jy出边界了！\n");
                int ridx = (ii + i) + (jj + j) * dim.x;
                eulerian_v[ridx] += f[4 * (jj + 1) + (ii + 1)];
            }
        }
    }
}

void distribute_s(const double* lagrange_s, double* eulerian_s, int num, int2 dim, double hx, double hy,
                  const double4* quadrature_rules) {
    for (int gidx = 0; gidx < num; gidx++) {
        double4 pos    = quadrature_rules[gidx];
        int     i      = floor((pos.x + 0.5 * hx) / hx);
        int     j      = floor((pos.y + 0.5 * hy) / hy);
        double  f      = lagrange_s[gidx];
        double  inv_h2 = 1.0 / hx / hy;

        for (int jj = j - 1; jj <= j + 2; jj++) {
            for (int ii = i - 1; ii <= i + 2; ii++) {
                if (!(jj >= 0 && jj < dim.y && ii >= 0 && ii < dim.x)) { continue; }
                int    widx   = ii + jj * dim.x;
                double weight = pos.w * inv_h2
                                * ibm_delta2(make_double2((pos.x + 0.5 * hx) / hx, (pos.y + 0.5 * hy) / hy),
                                             make_double2(ii, jj));
                eulerian_s[widx] += f * weight;
            }
        }
    }
}