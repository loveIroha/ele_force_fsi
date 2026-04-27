/**
 * @file vector_io.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2023/08/29
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef __VECTOR_IO_H__
#define __VECTOR_IO_H__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace IO {
template <typename T>
using vector2D = std::vector<std::vector<T>>;
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
void write_value(std::ostream& out, const T& value) {
    out << std::scientific << std::setprecision(20) << value << " ";
}

// 重载模板函数，处理 double3 类型
inline void write_value(std::ostream& out, const double3& value) {
    out << std::scientific << std::setprecision(20) << " " << value.x << " " << value.y << " " << value.z << " ";
}

template <typename T>
auto write_vector_1D(const std::vector<T>& data, const std::string& filename) {
    // 打开文件以写入数据
    std::ofstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    // 遍历二维向量并将数据写入文件
    for (const auto& value : data) {
        write_value(file, value);
    }

    // 关闭文件
    file.close();
    return 0;
}

inline bool read_value(std::istringstream& iss, double& value) {
    if (iss >> value) {
        return true;
    } else {
        return false;
    }
}
inline bool read_value(std::istringstream& iss, double3& value) {
    if (iss >> value.x && iss >> value.y && iss >> value.z) {
        return true;
    } else {
        return false;
    }
}
template <typename T>
auto read_vector_1D(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    std::string    line;
    std::vector<T> dataArray;

    if (std::getline(file, line)) {
        std::vector<T>     row;
        std::istringstream iss(line);
        // NOTE: 任何数据都先读入double，然后再转换为T类型
        T value;
        while (read_value(iss, value)) {
            dataArray.push_back(value);
        }
    } else {
        throw std::runtime_error("Error reading file.");
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

// 写入3D向量数据到文件
template <typename T>
void write_vector_3D(const std::vector<std::vector<std::vector<T>>>& data, const std::string& filename) {
    // 打开文件以写入数据
    std::ofstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    // 遍历三维向量并将数据写入文件
    for (const auto& plane : data) {
        for (const auto& row : plane) {
            for (const auto& value : row) {
                file << std::scientific << std::setprecision(20) << value << " ";
            }
            file << std::endl; // 在行末添加换行符
        }
        file << std::endl; // 在平面末尾添加一个额外的空行
    }

    // 关闭文件
    file.close();
}

// 从文件中读取3D向量数据
template <typename T>
std::vector<std::vector<std::vector<T>>> read_vector_3D(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Error opening file."); }

    std::vector<std::vector<std::vector<T>>> dataArray;
    std::string                              line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // 跳过空行
        }

        std::vector<std::vector<T>> plane;
        while (!line.empty()) {
            std::vector<T>     row;
            std::istringstream iss(line);
            double             value;

            while (iss >> value) {
                row.push_back(static_cast<T>(value));
            }

            plane.push_back(row);
            if (!std::getline(file, line)) { break; }
        }

        dataArray.push_back(plane);
    }

    // 关闭文件
    file.close();
    return dataArray;
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

template <typename T>
auto print_vector_3D(std::vector<std::vector<std::vector<T>>> data, int width = 8) {
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            for (size_t k = 0; k < data[i][j].size(); k++) {
                printf("%.4f  ", data[i][j][k]);
                // std::cout << std::setw(width) << std::scientific <<
                // data[i][j][k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
} // namespace IO

#endif