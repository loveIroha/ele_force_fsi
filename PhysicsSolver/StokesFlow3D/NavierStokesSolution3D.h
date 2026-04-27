/// @date 2023-10-08
/// @file NavierStokesSolution3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 将交错网格上的物理量插值到单元中心，或者将单元中心的数据插值到交错网格上
///
///

#ifndef __NAVIER_STOKES_SOLUTION_3D_H__
#define __NAVIER_STOKES_SOLUTION_3D_H__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

// CUDA
#include <helper_math.h>
#include <vector_types.h>

// NOTE: 这个函数会将交错网格上的数据插值到单元中心，在输出数据的时候会使用
template <int DIM, typename T, typename TV>
void stagger_to_center(
    // std::array<algebra::MultiArray<DIM, T>, DIM> &center,
    algebra::MultiArrayType<DIM, TV>& center, const algebra::MultiArrayType<DIM, T>& stagger_u,
    const algebra::MultiArrayType<DIM, T>& stagger_v, const algebra::MultiArrayType<DIM, T>& stagger_w,
    const std::array<int, DIM>& dimensions) {
    // if constexpr (DIM == 3)
    // {
    for (int i = 0; i < dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            for (int k = 0; k < dimensions[2]; k++)
            // {
            //     center[0][i + 1][j + 1][k + 1] = 0.5 * (stagger[0][i][j +
            //     1][k + 1] + stagger[0][i + 1][j + 1][k + 1]); center[1][i +
            //     1][j + 1][k + 1] = 0.5 * (stagger[1][i + 1][j][k + 1] +
            //     stagger[1][i + 1][j + 1][k
            //     + 1]); center[2][i + 1][j + 1][k + 1] = 0.5 * (stagger[2][i +
            //     1][j + 1][k] + stagger[2][i + 1][j + 1][k + 1]);
            // }
            {
                center[i + 1][j + 1][k + 1].x = 0.5 * (stagger_u[i + 1][j + 1][k + 1] + stagger_u[i + 2][j + 1][k + 1]);
                center[i + 1][j + 1][k + 1].y = 0.5 * (stagger_v[i + 1][j + 1][k + 1] + stagger_v[i + 1][j + 2][k + 1]);
                center[i + 1][j + 1][k + 1].z = 0.5 * (stagger_w[i + 1][j + 1][k + 1] + stagger_w[i + 1][j + 1][k + 2]);
            }
        }
    }
    // }
    // if constexpr (DIM == 2)
    // {
    //     for (int i = 0; i < dimensions[0]; i++)
    //     {
    //         for (int j = 0; j < dimensions[1]; j++)
    //         {
    //             center[0][i + 1][j + 1] = 0.5 * (stagger[0][i][j + 1] +
    //             stagger[0][i + 1][j + 1]); center[1][i + 1][j + 1] = 0.5 *
    //             (stagger[1][i + 1][j] + stagger[1][i + 1][j + 1]);
    //         }
    //     }
    // }
}

template <int DIM, typename T, typename TV>
[[deprecated("这个函数会将单元中心的数据拆分到交错网格上，一般情况下会使用")]] 
void center_to_stagger(algebra::MultiArrayType<DIM, T>& stagger_u, algebra::MultiArrayType<DIM, T>& stagger_v,
                       algebra::MultiArrayType<DIM, T>& stagger_w, const algebra::MultiArrayType<DIM, TV>& center,
                       const std::array<int, DIM>& dimensions) {
    // if constexpr (DIM == 3)
    // {
    for (int i = 0; i < dimensions[0] + 1; i++) {
        for (int j = 0; j < dimensions[1] + 1; j++) {
            for (int k = 0; k < dimensions[2] + 1; k++) {
                stagger_u[i + 1][j + 1][k + 1] = 0.5 * (center[i][j + 1][k + 1].x + center[i + 1][j + 1][k + 1].x);
                stagger_v[i + 1][j + 1][k + 1] = 0.5 * (center[i + 1][j][k + 1].y + center[i + 1][j + 1][k + 1].y);
                stagger_w[i + 1][j + 1][k + 1] = 0.5 * (center[i + 1][j + 1][k].z + center[i + 1][j + 1][k + 1].z);
            }
        }
    }

    // }
    // if constexpr (DIM == 2)
    // {
    // }
}

#endif // __NAVIER_STOKES_SOLUTION_3D_H__