/**
 * @file MultigridBase22.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2023-05-18
 *
 * @copyright Copyright (c) 2023  Ma Pengfei
 *
 */

#ifndef __MULTIGRID_BASE_2_H__
#define __MULTIGRID_BASE_2_H__
// my library
#include <AlgebraSolver/GpuVector.h>

// standard library
#include <type_traits>

// boost library
#include <boost/type_index.hpp>

// third party library
#include <io/loguru.hpp>

#include "BoundaryTypes.h"

// NOTE：使用inline避免链接错误。
inline double L0(double x) { return 1.0 - x; }
inline double L1(double x) { return x; }
inline double (*L[2])(double) = {L0, L1};

inline double test_linears(const double* f, double x) {
    double sum = 0.0;
    for (int i = 0; i < 2; i++) {
        sum += f[i] * L[i](x);
    }
    return sum;
}

namespace pangu {

template <typename VectorType, typename BCVectorType, Dimension DIM>
class MultigridBase2 {
  private:
    std::vector<VectorType>   multi_x; // Ax = b
    std::vector<VectorType>   multi_b;
    std::vector<VectorType>   multi_r; // r = b - Ax
    std::vector<VectorType>   multi_e;
    std::vector<BCVectorType> multi_bc;

  protected:
    int3    _dim;
    double3 _dh;

    // 基本类型推导
    using BT = std::decay_t<decltype(multi_bc[0].data()[0])>;
    using TV = std::decay_t<decltype(multi_x[0].data()[0])>;
    using T  = std::decay_t<decltype(flatten(multi_x[0]).data[0])>;

  public:
    // 多重网格求解器参数
    int num_levels; // 网格层数
    int max_iteration        = 50;
    int num_initsmooth       = 1;
    int num_presmooth        = 2;
    int num_aftersmooth      = 2;
    int num_exactsmooth      = 100; // 最粗层的最大迭代次数
    int num_smallest_unkowns = 5;   // 每个维度最小网格单元数

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
    MultigridBase2(int3 dim, double3 dh) : _dim(dim), _dh(dh) {
        if constexpr (DIM == DIM2) num_smallest_unkowns = num_smallest_unkowns * num_smallest_unkowns;
        if constexpr (DIM == DIM3)
            num_smallest_unkowns = num_smallest_unkowns * num_smallest_unkowns * num_smallest_unkowns;

        compute_multigrid_levels(_dim, _dh, level_dim, level_dh, num_levels);

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
            multi_x[i]  = VectorType(level_dim[i].x + 1);
            multi_e[i]  = VectorType(level_dim[i].x + 1);
            multi_r[i]  = VectorType(level_dim[i].x + 1);
            multi_b[i]  = VectorType(level_dim[i].x + 1);
            multi_bc[i] = BCVectorType(level_dim[i].x + 1);
        }
        useGPU = multi_x[0].use_gpu();
    };

    virtual void smooth(VectorType& x, const VectorType& b, const BCVectorType& bc, int3 dim, double3 dh, int n) = 0;
    virtual void compute_residuals(const VectorType& x, const VectorType& b, VectorType& r, const BCVectorType& bc,
                                   int3 dim, double3 dh)
        = 0;

    virtual ~MultigridBase2(){};

    void compute_multigrid_levels(int3 dim, double3 dh, std::vector<int3>& level_dim, std::vector<double3>& level_dh,
                                  int& num_levels) {
        printf("\n        Calculating multigrid levels......\n\n");

        num_levels = 0;
        level_dh.clear();
        level_dim.clear();

        while (true) {
            num_levels = num_levels + 1;
            level_dh.push_back(dh);
            level_dim.push_back(dim);
            printf("num_levels : %d, level_dh  : %.12e, %.12e, %.12e \n", num_levels, dh.x, dh.y, dh.z);
            printf("num_levels : %d, level_dim : %d, %d, %d \n", num_levels, dim.x, dim.y, dim.z);

            if (dim.x % 2 == 1 || dim.x % 2 == 1 || dim.x % 2 == 1) break;
            if (dim.x * dim.y * dim.z < 64) break;

            dim = make_int3(dim.x / 2, std::max(dim.y / 2, 1), std::max(dim.z / 2, 1));
            dh  = {dh.x * 2.0, dh.y * 2.0, dh.z * 2.0};
        }
        CHECK_F(num_levels >= 1, "the problem is too small for multigrid solver.");
    }

    void restrict(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        if constexpr (DIM == DIM1) restrict1D(fine, coarse, dim_fine, dim_coarse);
    }

    void prolongate(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        if constexpr (DIM == DIM1) prolongate1D(fine, coarse, dim_fine, dim_coarse);
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
        //   *|*|*|*|*|*
        //  *-|-*-|-*-|-*
        // *--|---*---|--*

        // 0 12 -> 1 34 -> 2 5
        // [i*2-1] [i*2]
        auto dim_coarse_local = int3{dim_coarse.x + 1, 0, 0};
        // auto dim_dine_local = int3{dim_fine.x+1,0,0};
        for (int i = 0; i < dim_coarse_local.x; i++) {
            // auto l = fine.data()[2 * i - 1];
            auto l           = i == 0 ? fine.data()[0] : fine.data()[2 * i - 1];
            auto r           = i == dim_coarse.x - 1 ? fine.data()[dim_coarse.x - 1] : fine.data()[2 * i];
            coarse.data()[i] = 0.5 * (l + r);
        }
    }

    void restrict2D(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {}

    void restrict3D(const VectorType& fine, VectorType& coarse, int3 dim_fine, int3 dim_coarse) {}

    void prolongate1D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {
        //    *|*|*|*|*|*|*|*|*|*
        //   *-|-*-|-*-|-*-|-*-|-*
        // *---|---*---|---*---|---*

        // 01->01, 12->23, 23->45, 34->67
        // [i,i+1] -> [2*i, 2*i+1]
        auto dim_coarse_local = int3{dim_coarse.x + 1, 0, 0};
        auto dim_dine_local   = int3{dim_fine.x + 1, 0, 0};
        for (int i = 0; i < dim_coarse_local.x; i++) {
            double f[2]            = {coarse.data()[i], coarse.data()[i + 1]};
            fine.data()[2 * i]     = test_linears(f, 1.0 / 3.0);
            fine.data()[2 * i + 1] = test_linears(f, 2.0 / 3.0);
        }
    }

    void prolongate2D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {}

    void prolongate3D(VectorType& fine, const VectorType& coarse, int3 dim_fine, int3 dim_coarse) {}
};
} // namespace pangu

#endif
