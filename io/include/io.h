/// @date 2023-12-07
/// @file io.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <fmt/core.h>
#include <io/ScopeProfiler.h>
#include <io/cxxopts.hpp>
#include <io/loguru.hpp>
#include <io/readVTK.h>
#include <io/readtxt.h>
#include <io/smtp.h>
#include <io/vector_io.h>
#include <io/writeVTK.h>
#include <tuple>

namespace {
std::string geometry_path(std::string path) {
    std::string geometry_root_path = NPUHEART_GEOMETRY_PATH;
    std::string result             = geometry_root_path + path;
    return result;
}
int copy_json_to(const std::string& copy_from, const std::string& copy_to) {

    if (!std::filesystem::exists(copy_to)) std::filesystem::create_directories(copy_to);

    std::filesystem::path filepath = copy_from;
    std::string           filename = filepath.filename().string();

    // 构建目标文件的路径
    std::string destinationFile = copy_to + filename;

    // 打开源文件和目标文件
    std::ifstream sourceStream(copy_from, std::ios::binary);
    std::ofstream destinationStream(destinationFile, std::ios::binary);
    std::cout << destinationFile << std::endl;

    // 检查文件是否成功打开
    if (!sourceStream.is_open()) { CHECK_F(false, "Failed to open source file: %s!", copy_from.c_str()); }
    if (!destinationStream.is_open()) { CHECK_F(false, "Failed to open destination file: %s!", copy_from.c_str()); }

    // 从源文件复制到目标文件
    destinationStream << sourceStream.rdbuf();

    // 关闭文件流
    sourceStream.close();
    destinationStream.close();
    return 0;
}
} // namespace
