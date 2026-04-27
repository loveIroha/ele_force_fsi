/// @date 2023-10-16
/// @file GridArray.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __GRIDARRAY_H__
#define __GRIDARRAY_H__

#include <assert.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
template <class T, int DIM, class enable = void>
class GridArray;

struct Padding {
    int left{0}, right{0}, top{0}, bottom{0};
};

template <class T, int DIM>
class GridArray<T, DIM, std::enable_if_t<DIM == 2>> {
  public:
    Padding                  padding;
    std::vector<T>           data;
    std::array<int, 2>       size_raw;
    const std::array<int, 2> size;

    GridArray(std::array<int, DIM> size) : size_raw(size), size(size) { data.resize(size[0] * size[1]); }

    void pad(int i, int j, int k, int l) {
        padding.left   = i;
        padding.right  = j;
        padding.bottom = k;
        padding.top    = l;
        size_raw[0]    = padding.left + padding.right + size[0];
        size_raw[1]    = padding.bottom + padding.top + size[1];
        auto temp      = data;
        data.resize(size_raw[0] * size_raw[1]);
        fill(T{});
        for (int j = 0; j < size[1]; j++)
            for (int i = 0; i < size[0]; i++)
                data[id(i, j)] = temp[id_raw(i, j)];
    }

    // The memory index of cell (i,j)
    inline int id(int i, int j) const { return (padding.left + i) + (padding.bottom + j) * size_raw[0]; }

    // The geometry index of cell (i,j)
    inline int id_raw(int i, int j) const { return i + j * size[0]; }

    void check_index(int i, int j) const {
        assert(i >= -padding.left && i < size[0] + padding.right);
        assert(j >= -padding.bottom && j < size[1] + padding.top);
    }

    T& operator()(int i, int j) {
        check_index(i, j);
        return data[id(i, j)];
    }

    const T& operator()(int i, int j) const {
        check_index(i, j);
        return data[id(i, j)];
    }

    void fill(const T& z) {
        for (int i = 0; i < size_raw[0] * size_raw[1]; i++)
            data[i] = z;
    }

    void fill_test() {
        for (int j = 0; j < size[1]; j++)
            for (int i = 0; i < size[0]; i++)
                data[j * size[0] + i] = j * 100 + i;
    }

    void print(std::ostream& out = std::cout, int width = 8) {
        for (int j = 0; j < size[1]; j++) {
            for (int i = 0; i < size[0]; i++) {
                out << std::setw(width)
                    // << data[id({i, j})] << " "
                    << data[id(i, j)] << " ";
            }
            out << std::endl;
        }
    }

    void print_raw(std::ostream& out = std::cout, int width = 12) {
        for (int j = -padding.bottom; j < size[1] + padding.top; j++) {
            for (int i = -padding.left; i < size[0] + padding.right; i++) {
                // std::cout << i << std::endl;
                out << std::setw(width)
                    // << data[id({i, j})] << " "
                    << data[id(i, j)] << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }
};

#endif // __GRIDARRAY_H__