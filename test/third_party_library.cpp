/// @date 2023-09-27
/// @file StokesFlow3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/StokesFlow3D/StokesFlow3D.h>
#include <fmt/core.h>
#include <mpParser.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <variant>

using json = nlohmann::json;

// 定义一个 JSON 数据的键值对类型
using JsonPair = std::pair<std::string, json>;
using JsonMap  = std::map<std::string, std::variant<int, double, bool, std::string>>;

int test_json() {
    try {
        // 从 JSON 文件中读取数据
        std::ifstream file("/home/fenics/npuheart-dev/input.json");
        if (!file.is_open()) { throw std::runtime_error("无法打开 JSON 文件"); }

        json jsonData;
        file >> jsonData;

        // 使用 std::map 存储 JSON 数据，值类型为 std::variant
        std::map<std::string, std::variant<int, double, bool, std::string>> dataMap;

        // std::map<std::string, std::variant<int, double, bool, std::string,
        // JsonMap>> dataMap;

        // 遍历 JSON 对象并填充映射
        for (const auto& pair : jsonData.items()) {
            const std::string& key   = pair.key();
            const json&        value = pair.value();

            // 判断 JSON 值的类型，并将其存储为 std::variant
            if (value.is_number_integer()) {
                dataMap[key] = value.get<int>();
            } else if (value.is_number_float()) {
                dataMap[key] = value.get<double>();
            } else if (value.is_boolean()) {
                dataMap[key] = value.get<bool>();
            } else if (value.is_string()) {
                dataMap[key] = value.get<std::string>();
            } else if (value.is_object()) {
                // 如果值是对象（嵌套 JSON），递归调用处理 JSON 对象
                std::cout << "嵌套 JSON" << std::endl;
            }
        }

        // 打印映射内容
        for (const auto& pair : dataMap) {
            const std::string&                                  key   = pair.first;
            const std::variant<int, double, bool, std::string>& value = pair.second;

            std::cout << "Key: " << key << ", Value: ";

            // 根据 std::variant 中的类型进行处理
            if (std::holds_alternative<int>(value)) {
                std::cout << std::get<int>(value);
            } else if (std::holds_alternative<double>(value)) {
                std::cout << std::get<double>(value);
            } else if (std::holds_alternative<bool>(value)) {
                std::cout << std::get<bool>(value);
            } else if (std::holds_alternative<std::string>(value)) {
                std::cout << std::get<std::string>(value);
            }

            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main() {
    double dt = 0.01;
    double T  = 1.0;
    size_t Nt = std::ceil(T / dt);

    size_t Nx = 4;
    size_t Ny = 4;
    size_t Nz = 4;

    double Lx = 1.0;
    double Ly = 1.0;
    double Lz = 1.0;

    double rho = 1.0;
    double mu  = 0.1;

    // StokesFlow<2> stokes(dt, Nx, Ny, Lx, Ly, T, rho, mu);

    StokesFlow<3> stokes(Nt, {Nx, Ny, Nz}, {Lx, Ly, Lz}, T, rho, mu);
    StokesFlow<2> stokes_2d(Nt, {Nx, Ny}, {Lx, Ly}, T, rho, mu);
    test_json();

    std::string msg = fmt::format("The answer is {}.\n", 42);
    std::cout << msg << std::endl;

    return 0;
}

// g++ -std=c++17
