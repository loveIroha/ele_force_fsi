/**
 * @file MultigridBase.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-10
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

// started at 2022-01-10
// finished at 2022-01-13

// NOTE: 2023-03-20 发现多重网格方法的边界条件尚未完全调通。

// restrict including restrict1D restrict2D restrict3D;
// prolong  including prolong1D  prolong2D  prolong3D;

// a simple explanation for restricion and prolongation operator.
// http://talk.pengfeima.cn/t/topic/80/7

// NOTE: When dim.x dim.y dim.z = 2^n+1, the convergence is good.

// 在smooth中施加边界条件，在残差求解时，只计算内点的残差。
// 限制算子和延拓算子参考了 Ulrich Trottenberg 写的教材《Multigrid》

#ifndef __MULTIGRID_BASE_H__
#define __MULTIGRID_BASE_H__
// my library
#include <AlgebraSolver/GpuVector.h>

// standard library
#include <type_traits>

// boost library
#include <boost/type_index.hpp>

// third party library
#include <io/loguru.hpp>

#include "BoundaryTypes.h"

namespace pangu {

// FIXME: There might be some problems about mix_boundary.
template <typename T>
T mix_boundary(T a1, T a2, T a3) {
    if constexpr (sizeof(T) / sizeof(char) == 1) {
        auto i = min(min(int(a1), int(a2)), int(a3));
        return char(i);
    }

    else if constexpr (sizeof(T) / sizeof(char) == 2) {
        auto i = min(min(toInt(a1), toInt(a2)), toInt(a3));
        return make_char2(i.x, i.y);
    }

    else if constexpr (sizeof(T) / sizeof(char) == 3) {
        auto i = min(min(toInt(a1), toInt(a2)), toInt(a3));
        return make_char3(i.x, i.y, i.z);
    }

    else if constexpr (sizeof(T) / sizeof(char) == 4) {
        auto i = min(min(toInt(a1), toInt(a2)), toInt(a3));
        return make_char4(i.x, i.y, i.z, i.w);
    }

    else {
        CHECK_F(false, "Wrong boundary type.");
        return T();
    }
}

// A example of usage:
// MultigridBase<StdVector<double,double3>, DIM1> gmg1;

template <typename VectorType, typename BCVectorType, Dimension DIM>
class MultigridBase {
    // TODO: keep it private.
  public:
    std::vector<BCVectorType> multi_bc;

  private:
    std::vector<VectorType> multi_x; // Ax = b
    std::vector<VectorType> multi_b;
    std::vector<VectorType> multi_r; // r = b - Ax
    std::vector<VectorType> multi_e;

    // Boundary type
    using BT = std::decay_t<decltype(multi_bc[0].data()[0])>;

  protected:
    int3    _dim;
    double3 _dh;

    using TV = std::decay_t<decltype(multi_x[0].data()[0])>;
    using T  = std::decay_t<decltype(flatten(multi_x[0]).data[0])>;

  public:
    // some solver parameters
    int max_iteration   = 50;
    int num_initsmooth  = 1;
    int num_presmooth   = 2;
    int num_aftersmooth = 2;
    // it should depend on the dimension of problem.
    int num_exactsmooth      = 100;
    int num_smallest_unkowns = 5;
    int num_levels;

    double tolerence = 1e-6;
    double omega     = 2.0 / 3.0; // param for SOR relaxation.

    bool useGPU = false;

  public:
    std::vector<int3>    level_dim;
    std::vector<double3> level_dh;

    // Check type name T and TV
    template <typename Type>
    void check_typename(Type tmprv) {
        using boost::typeindex::type_id_with_cvr;
        std::cout << "Type=" << type_id_with_cvr<Type>().pretty_name() << std::endl;
    }
    void check_TV() {
        TV b;
        check_typename(b);
    }

  public:
    void print_bcs(BCVectorType bcs, int3 dim) {
        for (size_t k = 0; k < dim.z; k++) {
            for (size_t j = 0; j < dim.y; j++) {
                for (size_t i = 0; i < dim.x; i++) {
                    for (size_t n = 0; n < bcs.value_size(); n++) {
                        printf("%d", ((char*)&bcs.data()[i + dim.x * j + k * dim.x * dim.y])[n]);
                    }
                    printf(", ");
                }
                printf("\n");
            }
            printf("\n\n\n\n");
        }
    }

    void print_bcs_2D(int n) {
        auto         dim = level_dim[n];
        BCVectorType bcs = multi_bc[n];
        for (int j = 0; j < dim.y; j++) {
            for (int i = 0; i < dim.x; i++) {
                for (size_t n = 0; n < bcs.value_size(); n++) {
                    printf("%d", ((char*)&bcs.data()[i + dim.x * j])[n]);
                }
                printf(", ");
            }
            printf("\n");
        }
        printf("end\n");
    }

    MultigridBase(int3 dim, double3 dh) : _dim(dim), _dh(dh) {
        if constexpr (DIM == DIM2) num_smallest_unkowns = num_smallest_unkowns * num_smallest_unkowns;
        if constexpr (DIM == DIM3)
            num_smallest_unkowns = num_smallest_unkowns * num_smallest_unkowns * num_smallest_unkowns;

        compute_multigrid_levels(_dim, level_dim, level_dh, num_levels);

        // NOTE : if the capacity of vector is not enough, the original vector
        // will be destroyed and data will be copied to another place. the copy
        // constructor will be called. which is very expensive.
        multi_x.resize(num_levels);
        multi_e.resize(num_levels);
        multi_r.resize(num_levels);
        multi_b.resize(num_levels);
        multi_bc.resize(num_levels);
        for (int i = 0; i < num_levels; i++) {
            // std::cout << "\n\n    Allocating memory " <<
            // level_dim[i].x*level_dim[i].y*level_dim[i].z << "
            // ......\n\n";
            multi_x[i]  = VectorType(level_dim[i]);
            multi_e[i]  = VectorType(level_dim[i]);
            multi_r[i]  = VectorType(level_dim[i]);
            multi_b[i]  = VectorType(level_dim[i]);
            multi_bc[i] = BCVectorType(level_dim[i]);
        }
        useGPU = multi_x[0].use_gpu();
    };

    void calculate_bcs(const BCVectorType& bcs) {
        CHECK_F(bcs.use_gpu() == useGPU, "This is a GPU version of MultigridBase. Construct multilevel BC with a CPU "
                                         "version of MultigridBase, copy that.");
        multi_bc[0] = bcs;
        for (int i = 0; i < num_levels - 1; i++) {
            restrict_bcs(multi_bc[i], multi_bc[i + 1], level_dim[i], level_dim[i + 1]);
        }
    }

    void restrict_bcs(const BCVectorType& fine, BCVectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        if constexpr (DIM == DIM1) restrict_bcs1D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM2) restrict_bcs2D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM3) restrict_bcs3D(fine, coarse, dim_fine, dim_coarse);
    }

    void restrict_bcs1D(const BCVectorType& fine, BCVectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        /// 1D inner points.
        for (int i = 1; i < dim_coarse.x - 1; i++) {
            coarse.data()[i] = mix_boundary(fine.data()[2 * i - 1], fine.data()[2 * i], fine.data()[2 * i + 1]);
        }

        coarse.data()[0]                = fine.data()[0];
        coarse.data()[dim_coarse.x - 1] = fine.data()[dim_fine.x - 1];
    }

    void restrict_bcs2D(const BCVectorType& fine, BCVectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        // Default boundary conditions are Dirichlet type, four corners for the 2D
        // situation.
        coarse = DIRICHLET;

        // 2D inner points.
        auto c = coarse.data();
        auto f = fine.data();
        for (int i = 1; i < dim_coarse.x - 1; i++) {
            for (int j = 1; j < dim_coarse.y - 1; j++) {
                auto ii = 2 * i;
                auto jj = 2 * j;
                auto c1 = mix_boundary(f[ii - 1 + (jj - 1) * dim_fine.x], f[ii + 1 + (jj - 1) * dim_fine.x],
                                       f[ii - 1 + (jj + 1) * dim_fine.x]);
                auto c2 = mix_boundary(f[ii + 1 + (jj + 1) * dim_fine.x], f[ii + 1 + jj * dim_fine.x],
                                       f[ii - 1 + jj * dim_fine.x]);
                auto c3 = mix_boundary(f[ii + (jj + 1) * dim_fine.x], f[ii + (jj - 1) * dim_fine.x],
                                       f[ii + jj * dim_fine.x]);
                c[i + j * dim_coarse.x] = mix_boundary(c1, c2, c3);
            }
        }

        for (int i = 1; i < dim_coarse.x - 1; i++) {
            auto ii = 2 * i;
            // down
            int  j  = 0;
            auto jj = 2 * j;
            c[i + j * dim_coarse.x]
                = mix_boundary(f[ii - 1 + jj * dim_fine.x], f[ii + jj * dim_fine.x], f[ii + 1 + jj * dim_fine.x]);
            // top
            j  = dim_coarse.y - 1;
            jj = 2 * j;
            c[i + j * dim_coarse.x]
                = mix_boundary(f[ii - 1 + jj * dim_fine.x], f[ii + jj * dim_fine.x], f[ii + 1 + jj * dim_fine.x]);
        }
        for (int j = 1; j < dim_coarse.y - 1; j++) {
            auto jj = 2 * j;
            // left
            int  i  = 0;
            auto ii = 2 * i;
            c[i + j * dim_coarse.x]
                = mix_boundary(f[ii + (jj - 1) * dim_fine.x], f[ii + jj * dim_fine.x], f[ii + (jj + 1) * dim_fine.x]);
            // right
            i  = dim_coarse.x - 1;
            ii = 2 * i;
            c[i + j * dim_coarse.x]
                = mix_boundary(f[ii + (jj - 1) * dim_fine.x], f[ii + jj * dim_fine.x], f[ii + (jj + 1) * dim_fine.x]);
        }
    }

    void restrict_bcs3D(const BCVectorType& fine, BCVectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        // Default boundary conditions are Dirichlet type, eight corners and
        // twelve edges for the 3D situation.
        coarse = DIRICHLET;

        // 3D inner points.
        auto c = coarse.data();
        auto f = fine.data();

        for (int i = 1; i < dim_coarse.x - 1; i++) {
            for (int j = 1; j < dim_coarse.y - 1; j++) {
                for (int k = 1; k < dim_coarse.z - 1; k++) {
                    auto ii = 2 * i;
                    auto jj = 2 * j;
                    auto kk = 2 * k;

                    c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(
                        mix_boundary(
                            mix_boundary(f[ii - 1 + (jj - 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                         f[ii - 1 + (jj - 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                         f[ii - 1 + (jj + 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y]),
                            mix_boundary(f[ii - 1 + (jj + 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                         f[ii + 1 + (jj - 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                         f[ii + 1 + (jj - 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y]),
                            mix_boundary(f[ii + 1 + (jj + 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                         f[ii + 1 + (jj + 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                         f[ii + 1 + (jj - 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y])),
                        mix_boundary(mix_boundary(f[ii + 1 + (jj + 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y],
                                                  f[ii - 1 + (jj - 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y],
                                                  f[ii - 1 + (jj + 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y]),
                                     mix_boundary(f[ii + 1 + (jj)*dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + 1 + (jj)*dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                                  f[ii - 1 + (jj)*dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y]),
                                     mix_boundary(f[ii - 1 + (jj)*dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + (jj + 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + (jj + 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y])),
                        mix_boundary(mix_boundary(f[ii + (jj - 1) * dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + (jj - 1) * dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + (jj)*dim_fine.x + (kk + 1) * dim_fine.x * dim_fine.y]),
                                     mix_boundary(f[ii + (jj)*dim_fine.x + (kk - 1) * dim_fine.x * dim_fine.y],
                                                  f[ii + (jj + 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y],
                                                  f[ii + (jj - 1) * dim_fine.x + (kk)*dim_fine.x * dim_fine.y]),
                                     mix_boundary(f[ii + 1 + (jj)*dim_fine.x + (kk)*dim_fine.x * dim_fine.y],
                                                  f[ii - 1 + (jj)*dim_fine.x + (kk)*dim_fine.x * dim_fine.y],
                                                  f[ii + jj * dim_fine.x + kk * dim_fine.x * dim_fine.y])));
                }
            }
        }

        // six faces.
        int stride_z = dim_fine.x * dim_fine.y;
        for (int i = 1; i < dim_coarse.x - 1; i++) {
            for (int j = 1; j < dim_coarse.y - 1; j++) {
                int ii = 2 * i;
                int jj = 2 * j;
                // front
                int  k  = 0;
                int  kk = 2 * k;
                auto c1 = mix_boundary(f[ii - 1 + (jj - 1) * dim_fine.x + kk * stride_z],
                                       f[ii + 1 + (jj - 1) * dim_fine.x + kk * stride_z],
                                       f[ii - 1 + (jj + 1) * dim_fine.x + kk * stride_z]);
                auto c2 = mix_boundary(f[ii + 1 + (jj + 1) * dim_fine.x + kk * stride_z],
                                       f[ii + 1 + jj * dim_fine.x + kk * stride_z],
                                       f[ii - 1 + jj * dim_fine.x + kk * stride_z]);
                auto c3 = mix_boundary(f[ii + (jj + 1) * dim_fine.x + kk * stride_z],
                                       f[ii + (jj - 1) * dim_fine.x + kk * stride_z],
                                       f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);

                // back
                k  = dim_coarse.z - 1;
                kk = 2 * k;
                c1 = mix_boundary(f[ii - 1 + (jj - 1) * dim_fine.x + kk * stride_z],
                                  f[ii + 1 + (jj - 1) * dim_fine.x + kk * stride_z],
                                  f[ii - 1 + (jj + 1) * dim_fine.x + kk * stride_z]);
                c2 = mix_boundary(f[ii + 1 + (jj + 1) * dim_fine.x + kk * stride_z],
                                  f[ii + 1 + jj * dim_fine.x + kk * stride_z],
                                  f[ii - 1 + jj * dim_fine.x + kk * stride_z]);
                c3 = mix_boundary(f[ii + (jj + 1) * dim_fine.x + kk * stride_z],
                                  f[ii + (jj - 1) * dim_fine.x + kk * stride_z],
                                  f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);
            }
        }

        for (int i = 1; i < dim_coarse.x - 1; i++) {
            for (int k = 1; k < dim_coarse.z - 1; k++) {
                int ii = 2 * i;
                int kk = 2 * k;
                // down
                int  j  = 0;
                int  jj = 2 * j;
                auto c1 = mix_boundary(f[ii - 1 + jj * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii + 1 + jj * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii - 1 + jj * dim_fine.x + (kk + 1) * stride_z]);
                auto c2 = mix_boundary(f[ii + 1 + jj * dim_fine.x + (kk + 1) * stride_z],
                                       f[ii + 1 + jj * dim_fine.x + kk * stride_z],
                                       f[ii - 1 + jj * dim_fine.x + kk * stride_z]);
                auto c3 = mix_boundary(f[ii + jj * dim_fine.x + (kk + 1) * stride_z],
                                       f[ii + jj * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);
                // up
                j  = dim_coarse.y - 1;
                jj = 2 * j;
                c1 = mix_boundary(f[ii - 1 + jj * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii + 1 + jj * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii - 1 + jj * dim_fine.x + (kk + 1) * stride_z]);
                c2 = mix_boundary(f[ii + 1 + jj * dim_fine.x + (kk + 1) * stride_z],
                                  f[ii + 1 + jj * dim_fine.x + kk * stride_z],
                                  f[ii - 1 + jj * dim_fine.x + kk * stride_z]);
                c3 = mix_boundary(f[ii + jj * dim_fine.x + (kk + 1) * stride_z],
                                  f[ii + jj * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);
            }
        }

        for (int j = 1; j < dim_coarse.y - 1; j++) {
            for (int k = 1; k < dim_coarse.z - 1; k++) {
                int jj = 2 * j;
                int kk = 2 * k;
                // left
                int  i  = 0;
                int  ii = 2 * i;
                auto c1 = mix_boundary(f[ii + (jj - 1) * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii + (jj + 1) * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii + (jj - 1) * dim_fine.x + (kk + 1) * stride_z]);
                auto c2 = mix_boundary(f[ii + (jj + 1) * dim_fine.x + (kk + 1) * stride_z],
                                       f[ii + (jj + 1) * dim_fine.x + kk * stride_z],
                                       f[ii + (jj - 1) * dim_fine.x + kk * stride_z]);
                auto c3 = mix_boundary(f[ii + jj * dim_fine.x + (kk + 1) * stride_z],
                                       f[ii + jj * dim_fine.x + (kk - 1) * stride_z],
                                       f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);
                // right
                i  = dim_coarse.x - 1;
                ii = 2 * i;
                c1 = mix_boundary(f[ii + (jj - 1) * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii + (jj + 1) * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii + (jj - 1) * dim_fine.x + (kk + 1) * stride_z]);
                c2 = mix_boundary(f[ii + (jj + 1) * dim_fine.x + (kk + 1) * stride_z],
                                  f[ii + (jj + 1) * dim_fine.x + kk * stride_z],
                                  f[ii + (jj - 1) * dim_fine.x + kk * stride_z]);
                c3 = mix_boundary(f[ii + jj * dim_fine.x + (kk + 1) * stride_z],
                                  f[ii + jj * dim_fine.x + (kk - 1) * stride_z],
                                  f[ii + jj * dim_fine.x + kk * stride_z]);
                c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y] = mix_boundary(c1, c2, c3);
            }
        }
    }

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh, int n) = 0;
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh)
        = 0;

    virtual ~MultigridBase(){};

    void compute_multigrid_levels(int3 dim, std::vector<int3>& level_dim, std::vector<double3>& level_dh,
                                  int& num_levels) {
        printf("\n        Calculating multigrid levels......\n\n");
        num_levels = 0;
        level_dim.clear();
        double3 dh = _dh;

        while (dim.x * dim.y * dim.z > num_smallest_unkowns) {
            level_dh.push_back(dh);
            level_dim.push_back(dim);

            printf("num_levels : %d, level_dh  : %.12e, %.12e, %.12e \n", num_levels, dh.x, dh.y, dh.z);
            printf("num_levels : %d, level_dim : %d, %d, %d \n", num_levels, dim.x, dim.y, dim.z);

            if constexpr (DIM == DIM1) dim = make_int3((dim.x + 1) / 2, 1, 1);
            if constexpr (DIM == DIM2) dim = make_int3((dim.x + 1) / 2, (dim.y + 1) / 2, 1);
            if constexpr (DIM == DIM3) dim = make_int3((dim.x + 1) / 2, (dim.y + 1) / 2, (dim.z + 1) / 2);

            dh         = {dh.x * 2.0, dh.y * 2.0, dh.z * 2.0};
            num_levels = num_levels + 1;
        }
        CHECK_F(num_levels >= 1, "the problem is too small for multigrid solver.");
    }

    void restrict(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        if constexpr (DIM == DIM1) restrict1D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM2) restrict2D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM3) {
            if (!useGPU) restrict3D(fine, coarse, dim_fine, dim_coarse);
            if (useGPU) gpu::restrict3D<T, TV>(fine.data(), coarse.data(), dim_fine, dim_coarse);
        }
    }

    void prolongate(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        if constexpr (DIM == DIM1) prolongate1D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM2) prolongate2D(fine, coarse, dim_fine, dim_coarse);
        if constexpr (DIM == DIM3) {
            if (!useGPU) prolongate3D(fine, coarse, dim_fine, dim_coarse);
            if (useGPU) gpu::prolongate3D<T, TV>(fine.data(), coarse.data(), dim_fine, dim_coarse);
        }
    }

    void correct_errors(VectorType& x, const VectorType& e) { x.axpy(1.0, x, e); }

    void iterate(VectorType& x, const VectorType& b, int current_level, double omega = 2.0 / 3.0, int a1 = 2,
                 int a2 = 1, int v = 1) {
        CHECK_F(x.size() == b.size(), "WRONG size.");

        if (current_level == 0) {
            multi_x[0] = x;
            multi_b[0] = b;
        }

        for (int i = 0; i < num_presmooth; i++) {
            smooth(multi_x[current_level], multi_b[current_level], multi_bc[current_level], level_dim[current_level],
                   level_dh[current_level], 1);
        }

        if (current_level < num_levels - 1) {
            for (int k = 0; k < v; k++) {
                // zero(i);
                multi_x[current_level + 1] = 0.0;

                // Compute residuals r = b - Ax
                compute_residuals(multi_x[current_level], multi_b[current_level], multi_r[current_level],
                                  multi_bc[current_level], level_dim[current_level], level_dh[current_level]);

                // Restrict residuals
                restrict(multi_r[current_level], multi_b[current_level + 1], level_dim[current_level],
                         level_dim[current_level + 1]);

                // Solve residuals Ae = r
                iterate(multi_x[current_level + 1], multi_b[current_level + 1], current_level + 1);

                // Prolongate residuals
                prolongate(multi_e[current_level], multi_x[current_level + 1], level_dim[current_level],
                           level_dim[current_level + 1]);

                // Correct errors x = x + e
                correct_errors(multi_x[current_level], multi_e[current_level]);
            }
        } else {
            for (int i = 0; i < num_exactsmooth; i++) {
                smooth(multi_x[current_level], multi_b[current_level], multi_bc[current_level],
                       level_dim[current_level], level_dh[current_level], 1);
            }
        }

        // Post smooth
        for (int i = 0; i < num_aftersmooth; i++) {
            smooth(multi_x[current_level], multi_b[current_level], multi_bc[current_level], level_dim[current_level],
                   level_dh[current_level], 1);
        }

        if (current_level == 0) x = multi_x[0];
    }

  private:
    void restrict1D(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        for (int i = 0; i < dim_coarse.x; i++) {
            auto l           = 2 * i - 1 < 0 ? fine.data()[2 * i + 1] : fine.data()[2 * i - 1];
            auto r           = 2 * i + 1 > dim_fine.x - 1 ? fine.data()[2 * i - 1] : fine.data()[2 * i + 1];
            auto m           = fine.data()[2 * i];
            coarse.data()[i] = 0.25 * (l + 2.0 * m + r);
        }
    }

    void restrict2D(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        auto c = coarse.data();
        auto f = fine.data();
        for (int i = 0; i < dim_coarse.x; i++) {
            for (int j = 0; j < dim_coarse.y; j++) {
                auto ii = 2 * i;
                auto jj = 2 * j;

                auto rr = (ii + 1 > dim_fine.x - 1) ? f[ii - 1 + jj * dim_fine.x] : f[ii + 1 + jj * dim_fine.x];
                auto ll = (ii - 1 < 0) ? f[ii + 1 + jj * dim_fine.x] : f[ii - 1 + jj * dim_fine.x];
                auto dd = (jj + 1 > dim_fine.y - 1) ? f[ii + (jj - 1) * dim_fine.x] : f[ii + (jj + 1) * dim_fine.x];
                auto uu = (jj - 1 < 0) ? f[ii + (jj + 1) * dim_fine.x] : f[ii + (jj - 1) * dim_fine.x];
                auto mm = f[ii + jj * dim_fine.x];

                c[i + j * dim_coarse.x] = 0.125 * (rr + ll + uu + dd) + 0.5 * mm;
            }
        }
    }

    void restrict3D(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        auto c = coarse.data();
        auto f = fine.data();
        for (int i = 0; i < dim_coarse.x; i++) {
            for (int j = 0; j < dim_coarse.y; j++) {
                for (int k = 0; k < dim_coarse.z; k++) {
                    auto ii = 2 * i;
                    auto jj = 2 * j;
                    auto kk = 2 * k;

                    auto ystride     = dim_fine.x;
                    auto zstride     = dim_fine.x * dim_fine.y;
                    auto centerindex = ii + jj * ystride + kk * zstride;

                    TV ll = (ii == 0) ? f[centerindex + 1] : f[centerindex - 1];
                    TV rr = (ii == dim_fine.x - 1) ? f[centerindex - 1] : f[centerindex + 1];
                    TV dd = (jj == 0) ? f[centerindex + ystride] : f[centerindex - ystride];
                    TV uu = (jj == dim_fine.y - 1) ? f[centerindex - ystride] : f[centerindex + ystride];
                    TV ff = (kk == 0) ? f[centerindex + zstride] : f[centerindex - zstride];
                    TV bb = (kk == dim_fine.z - 1) ? f[centerindex - zstride] : f[centerindex + zstride];
                    TV mm = f[centerindex];
                    c[i + j * dim_coarse.x + k * dim_coarse.x * dim_coarse.y]
                        = (ll + rr + dd + uu + ff + bb) / 12.0 + mm * 0.5;
                }
            }
        }
    }

    void prolongate1D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        for (int i = 0; i < dim_fine.x; i++) {
            double u  = 0.5 * ((i) % 2);
            int    ii = i / 2 + 1;

            auto l = coarse.data()[ii - 1];
            auto r = ii > dim_coarse.x - 1 ? coarse.data()[ii - 1] : coarse.data()[ii];

            fine.data()[i] = (1.0 - u) * l + u * r;
        }
    }

    void prolongate2D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        auto c = coarse.data();
        auto f = fine.data();

        for (int i = 0; i < dim_fine.x; i++)
            for (int j = 0; j < dim_fine.y; j++) {
                int ii = i / 2;
                int jj = j / 2;

                TV m = c[ii + jj * dim_coarse.x];
                TV r = (ii + 1 > dim_coarse.x - 1) ? c[ii - 1 + jj * dim_coarse.x] : c[ii + 1 + jj * dim_coarse.x];
                TV l = (ii - 1 < 0) ? c[ii + 1 + jj * dim_coarse.x] : c[ii - 1 + jj * dim_coarse.x];
                TV d = (jj + 1 > dim_coarse.y - 1) ? c[ii + (jj - 1) * dim_coarse.x] : c[ii + (jj + 1) * dim_coarse.x];
                TV u = (jj - 1 < 0) ? c[ii + (jj + 1) * dim_coarse.x] : c[ii + (jj - 1) * dim_coarse.x];

                if (i % 2 == 0 && j % 2 == 0) f[i + j * dim_fine.x] = m;
                if (i % 2 == 1 && j % 2 == 0) f[i + j * dim_fine.x] = 0.5 * (l + r);
                if (i % 2 == 0 && j % 2 == 1) f[i + j * dim_fine.x] = 0.5 * (d + u);
                if (i % 2 == 1 && j % 2 == 1) f[i + j * dim_fine.x] = 0.25 * (l + r + d + u);
            }
    }

    // TODO : change it to trilinear interpolation.
    void prolongate3D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        auto c = coarse.data();
        auto f = fine.data();

        for (int i = 0; i < dim_fine.x; i++)
            for (int j = 0; j < dim_fine.y; j++)
                for (int k = 0; k < dim_fine.z; k++) {
                    // int ii = i/2;
                    // int jj = j/2;
                    // int kk = k/2;

                    // TV ll = (ii==0)                               ?
                    // f[centerindex+1]       : f[centerindex-1]; TV rr =
                    // (ii==dim_fine.x - 1)                  ? f[centerindex-1] :
                    // f[centerindex+1]; TV dd = (jj==0) ? f[centerindex+ystride] :
                    // f[centerindex-ystride]; TV uu = (jj==dim_fine.y - 1) ?
                    // f[centerindex-ystride] : f[centerindex+ystride]; TV ff =
                    // (kk==0)                               ? f[centerindex+zstride]
                    // : f[centerindex-zstride]; TV bb = (kk==dim_fine.z - 1) ?
                    // f[centerindex-zstride] : f[centerindex+zstride]; TV mm =
                    // f[centerindex];

                    T u = 0.5 * (i % 2);
                    T w = 0.5 * (j % 2);
                    T v = 0.5 * (k % 2);

                    int ii = i / 2 + 1;
                    int jj = j / 2 + 1;
                    int kk = k / 2 + 1;

                    auto centerindex = ii + jj * dim_coarse.x + kk * dim_coarse.x * dim_coarse.y;

                    auto ystride_coarse = dim_coarse.x;
                    auto zstride_coarse = dim_coarse.x * dim_coarse.y;

                    auto data0 = c[centerindex - 1 - ystride_coarse - zstride_coarse];
                    auto data1 = ii > dim_coarse.x - 1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                                       : c[centerindex - ystride_coarse - zstride_coarse];
                    auto data2 = jj > dim_coarse.y - 1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                                       : c[centerindex - 1 - zstride_coarse];
                    auto data3 = (ii > dim_coarse.x - 1 || jj > dim_coarse.y - 1)
                                     ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                     : c[centerindex - zstride_coarse];

                    auto data4 = kk > dim_coarse.z - 1 ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                                       : c[centerindex - 1 - ystride_coarse];
                    auto data5 = (kk > dim_coarse.z - 1 || ii > dim_coarse.x - 1)
                                     ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                     : c[centerindex - ystride_coarse];
                    auto data6 = (kk > dim_coarse.z - 1 || jj > dim_coarse.y - 1)
                                     ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                     : c[centerindex - 1];
                    auto data7 = (kk > dim_coarse.z - 1 || ii > dim_coarse.x - 1 || jj > dim_coarse.y - 1)
                                     ? c[centerindex - 1 - ystride_coarse - zstride_coarse]
                                     : c[centerindex];

                    T a[] = {(1.0 - u) * (1.0 - w) * (1.0 - v), (1.0 - w) * u * (1.0 - v), (1.0 - u) * w * (1.0 - v),
                             w * u * (1.0 - v)};
                    T b[] = {(1.0 - u) * (1.0 - w) * v, (1.0 - w) * u * v, (1.0 - u) * w * v, w * u * v};

                    TV e[] = {data0, data1, data2, data3};
                    TV d[] = {data4, data5, data6, data7};

                    // dot(e,a)+dot(d,b)
                    f[i + j * dim_fine.x + k * dim_fine.x * dim_fine.y] = a[0] * e[0] + a[1] * e[1] + a[2] * e[2]
                                                                          + a[3] * e[3] + b[0] * d[0] + b[1] * d[1]
                                                                          + b[2] * d[2] + b[3] * d[3];
                }
    }
};
} // namespace pangu

#endif
