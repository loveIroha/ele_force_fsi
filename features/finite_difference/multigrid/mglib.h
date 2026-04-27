/// @date 2023-08-12
/// @file mglib.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief function for multigrid solver
///
///

#ifndef __MGLIB_H__
#define __MGLIB_H__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

const int NEUMANN   = 2;
const int DIRICHLET = 1;

namespace mg {
template <typename T>
using vector2D = std::vector<std::vector<T>>;

struct LevelInfo {
    double dx;
    double dy;
    int    Nx;
    int    Ny;
};

template <typename T>
auto make_vector_2D(int N, int M) {
    std::vector<std::vector<T>> result;
    result.resize(N, std::vector<T>(M));
    return result;
}

template <typename T>
auto make_vector_3D(int N, int M, int L) {
    std::vector<std::vector<std::vector<T>>> result;
    result.resize(N, std::vector<std::vector<T>>(M, std::vector<T>(L)));
    return result;
}

template <typename T>
auto read_vector_2D(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    std::vector<std::vector<T>> dataArray;
    std::string                 line;

    while (std::getline(file, line)) {
        std::vector<T>     row;
        std::istringstream iss(line);
        // NOTE: 任何数据都先读入double，然后再转换为T类型
        double value;

        while (iss >> value) {
            row.push_back(value);
        }

        dataArray.push_back(row);
    }

    // Close the file
    file.close();
    return dataArray;
}

template <typename T>
auto write_vector_2D(const vector2D<T>& data, const std::string& filename) {
    // 打开文件以写入数据
    std::ofstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    // 遍历二维向量并将数据写入文件
    for (const auto& row : data) {
        for (const auto& value : row) {
            file << std::scientific << std::setprecision(20) << value << " ";
        }
        file << std::endl; // 在行末添加换行符
    }

    // 关闭文件
    file.close();
    return 0;
}

template <typename T>
auto print_vector_2D(std::vector<std::vector<T>> data, int width = 8) {
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            // std::cout << std::setw(width) << std::fixed << data[i][j];
            std::cout << std::setw(width) << std::scientific << data[i][j];
            // printf("%1d ", data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

std::tuple<int, std::vector<LevelInfo>> compute_multigrid_levels_2D(int Nx, int Ny, double width, double height);

vector2D<double> interpolate(const vector2D<double>& coarse, int Nx, int Ny);
vector2D<double> restrict(const vector2D<double>& fine, int Nx, int Ny);
vector2D<int> restrict_bc(const vector2D<int>& fine, int Nx, int Ny);

std::vector<std::vector<double>> smooth(std::vector<std::vector<double>>&       phi,
                                        const std::vector<std::vector<int>>&    boundary_type,
                                        const std::vector<std::vector<double>>& b, int Nx, int Ny, double width,
                                        double height);
std::vector<std::vector<double>> compute_residual(std::vector<std::vector<double>>&       phi,
                                                  const std::vector<std::vector<int>>&    boundary_type,
                                                  const std::vector<std::vector<double>>& b, int Nx, int Ny,
                                                  double width, double height);
} // namespace mg
#endif
