/// @date 2024-07-05
/// @file StaggeredGrid.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include <array>
#include <iostream>
#include <vector_types.h>

class StaggeredGrid {

  public:
    // 构造函数，初始化网格的extent, origin和spacing
    StaggeredGrid(const std::array<int, 3>& extent, const std::array<double, 3>& origin,
                  const std::array<double, 3>& spacing)
        : m_extent(extent), m_origin(origin), m_spacing(spacing) {}

    // 构造函数，初始化网格的extent, origin和spacing
    StaggeredGrid(const int3& extent, const double3& origin,
                  const double3& spacing)
        : m_extent{extent.x, extent.y, extent.z}, m_origin{origin.x,origin.y,origin.z}, m_spacing{spacing.z,spacing.y,spacing.z} {}

    enum class Offset { x, y, z };

    // 函数，输入(i, j, k)单元，偏置后的坐标
    //  (u,v,w) 分别往 x,y,z 方向的偏置
    template <Offset offset>
    std::array<double, 3> x(const std::array<int, 3>& index) const {
        std::array<double, 3> coordinate;
        coordinate[0] = m_origin[0] + (index[0] + 0.5) * m_spacing[0]; // x方向偏置0.5
        coordinate[1] = m_origin[1] + (index[1] + 0.5) * m_spacing[1];
        coordinate[2] = m_origin[2] + (index[2] + 0.5) * m_spacing[2];

        // 根据偏置方向调整相应的坐标
        if constexpr (offset == Offset::x) {
            coordinate[0] -= 0.5 * m_spacing[0];
        } else if constexpr (offset == Offset::y) {
            coordinate[1] -= 0.5 * m_spacing[1];
        } else if constexpr (offset == Offset::z) {
            coordinate[2] -= 0.5 * m_spacing[2];
        }

        return coordinate;
    }

    // 模板函数，输入坐标和偏置方向，返回对应的(i, j, k)索引
    template <Offset offset>
    std::array<int, 3> I(const std::array<double, 3>& coordinate) const {
        std::array<int, 3> index;
        double offset_coordinate[3] = {
            coordinate[0] - 0.5 * m_spacing[0], 
            coordinate[1] - 0.5 * m_spacing[1], 
            coordinate[2] - 0.5 * m_spacing[2]};

        // 根据偏置方向调整相应的坐标
        if constexpr (offset == Offset::x) {
            offset_coordinate[0] += 0.5 * m_spacing[0];
        } else if constexpr (offset == Offset::y) {
            offset_coordinate[1] += 0.5 * m_spacing[1];
        } else if constexpr (offset == Offset::z) {
            offset_coordinate[2] += 0.5 * m_spacing[2];
        }

        // 将坐标转换为索引
        index[0] = static_cast<int>(std::round((offset_coordinate[0] - m_origin[0]) / m_spacing[0]));
        index[1] = static_cast<int>(std::round((offset_coordinate[1] - m_origin[1]) / m_spacing[1]));
        index[2] = static_cast<int>(std::round((offset_coordinate[2] - m_origin[2]) / m_spacing[2]));

        return index;
    }

  private:
    std::array<int, 3> m_extent;  // 网格范围
    std::array<double, 3> m_origin;  // 网格原点
    std::array<double, 3> m_spacing; // 网格间距
};
