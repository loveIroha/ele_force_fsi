/// @date 2023-03-29
/// @file set_boundary_types.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 设置区域边界的类型和值
///
///

#ifndef _BOUNDARY_TYPES_
#define _BOUNDARY_TYPES_

#include <PhysicsSolver/multigrid/MultigridBase.h>
using namespace pangu;

template <typename T>
void set_one_value(int i, int j, int k, int3 dim, T* data, const T& value) {
    int xstride = 1;
    int ystride = dim.x;
    int zstride = dim.y * dim.z;
    int index   = i * xstride + j * ystride + k * zstride;
    data[index] = value;
}

template <typename T>
void set_twelve_edge_values(int3 dim, T* data, const T& value) {
    int i, j, k;
    for (i = 0; i < dim.x; i++) {
        j = 0;
        k = 0;
        set_one_value(i, j, k, dim, data, value);
        j = dim.y - 1;
        k = 0;
        set_one_value(i, j, k, dim, data, value);
        j = 0;
        k = dim.z - 1;
        set_one_value(i, j, k, dim, data, value);
        j = dim.y - 1;
        k = dim.z - 1;
        set_one_value(i, j, k, dim, data, value);
    }
    for (j = 0; j < dim.y; j++) {
        i = 0;
        k = 0;
        set_one_value(i, j, k, dim, data, value);
        i = dim.x - 1;
        k = 0;
        set_one_value(i, j, k, dim, data, value);
        i = 0;
        k = dim.z - 1;
        set_one_value(i, j, k, dim, data, value);
        i = dim.x - 1;
        k = dim.z - 1;
        set_one_value(i, j, k, dim, data, value);
    }
    for (k = 0; k < dim.z; k++) {
        i = 0;
        j = 0;
        set_one_value(i, j, k, dim, data, value);
        i = dim.x - 1;
        j = 0;
        set_one_value(i, j, k, dim, data, value);
        i = 0;
        j = dim.y - 1;
        set_one_value(i, j, k, dim, data, value);
        i = dim.x - 1;
        j = dim.y - 1;
        set_one_value(i, j, k, dim, data, value);
    }
}

template <typename T>
void set_six_face_values(int3 dim, T* data, const T& value) {
    int i, j, k;
    for (i = 0; i < dim.x; i++) {
        for (j = 0; j < dim.y; j++) {
            k = 0;
            set_one_value(i, j, k, dim, data, value);
            k = dim.z - 1;
            set_one_value(i, j, k, dim, data, value);
        }
    }
    for (i = 0; i < dim.x; i++) {
        for (k = 0; k < dim.z; k++) {
            j = 0;
            set_one_value(i, j, k, dim, data, value);
            j = dim.y - 1;
            set_one_value(i, j, k, dim, data, value);
        }
    }
    for (j = 0; j < dim.y; j++) {
        for (k = 0; k < dim.z; k++) {
            i = 0;
            set_one_value(i, j, k, dim, data, value);
            i = dim.x - 1;
            set_one_value(i, j, k, dim, data, value);
        }
    }
}
// 左右下上后前
template <typename T, typename TV>
void init_boundary_conditions(StdVector<T, TV>& x, StdVector<char, char>& bc, const int3& dim,
                              const char& left_type = NEUMANN, const TV& left_value = TV(),
                              const char& right_type = NEUMANN, const TV& right_value = TV(),
                              const char& bottom_type = NEUMANN, const TV& bottom_value = TV(),
                              const char& top_type = NEUMANN, const TV& top_value = TV(),
                              const char& back_type = NEUMANN, const TV& back_value = TV(),
                              const char& front_type = NEUMANN, const TV& front_value = TV()) {
    // for face x=0
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < dim.y; j++)
            for (int k = 0; k < dim.z; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.z]  = left_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = left_type;
            }

    // for face x=dim.x-1
    for (int i = dim.x - 1; i < dim.x; i++)
        for (int j = 0; j < dim.y; j++)
            for (int k = 0; k < dim.z; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.y]  = right_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = right_type;
            }

    // for face y=0
    for (int i = 0; i < dim.x; i++)
        for (int j = 0; j < 1; j++)
            for (int k = 0; k < dim.z; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.y]  = bottom_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = bottom_type;
            }

    // for face y=dim.y-1
    for (int i = 0; i < dim.x; i++)
        for (int j = dim.y - 1; j < dim.y; j++)
            for (int k = 0; k < dim.z; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.y]  = top_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = top_type;
            }

    // for face z=0
    for (int i = 0; i < dim.x; i++)
        for (int j = 0; j < dim.y; j++)
            for (int k = 0; k < 1; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.y]  = back_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = back_type;
            }

    // for face z=dim.z-1
    for (int i = 0; i < dim.x; i++)
        for (int j = 0; j < dim.y; j++)
            for (int k = dim.z - 1; k < dim.z; k++) {
                x.data()[i + j * dim.x + k * dim.x * dim.y]  = front_value;
                bc.data()[i + j * dim.x + k * dim.x * dim.z] = front_type;
            }

    set_twelve_edge_values(dim, x.data(), TV());
    set_twelve_edge_values(dim, bc.data(), DIRICHLET);
}

#endif