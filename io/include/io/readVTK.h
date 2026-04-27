/// @date 2024-07-04
/// @file readVTK.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
/// 
/// @brief 读取 vti 格式的文件
/// 
///

#pragma once
#include <io/writeVTK.h>

namespace IO {

// 可接受 char* 类型的参数。
template <typename T>
std::vector<T> parseStringToVector(const std::string& input) {
    std::vector<T>     vec;
    std::istringstream iss(input);
    std::string        token;

    while (iss >> token) {
        try {
            vec.push_back(std::stod(token)); // 将字符串转换为 double 并添加到向量中
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range& e) { std::cerr << "Out of range: " << e.what() << std::endl; }
    }
    return vec;
}

void split_string(const std::string& input, std::string& first_part, std::string& second_part, int n = 8) {
    CHECK_F(input.size() >= n, "base64 码必须大于等于 %d 个字符", n);
    first_part  = input.substr(0, n);
    second_part = input.substr(n);
}

template <typename TV>
auto base64_to_vector(const std::string& base64_sting) {
    // 将base64编码的字符串拆分为两部分，从头部解析出数据大小，从尾部解析出数据内容
    std::string first_part, second_part;
    split_string(base64_sting, first_part, second_part, 8);
    auto decoded         = base64_decode(first_part);
    auto raw_header      = reinterpret_cast<const std::uint32_t*>(decoded.data());
    auto length          = raw_header[0] / sizeof(TV);
    auto decoded_content = base64_decode(second_part);
    auto raw_content     = reinterpret_cast<const TV*>(decoded_content.data());
    CHECK_F(decoded_content.size() == raw_header[0], "解码后的数据长度必须和header所说的相等: %d ", raw_header[0]);
    std::vector<TV> vec(raw_content, raw_content + length);
    return vec;
}

auto read_vtk(const std::string& path) {
    struct VTIContent {
        int3                 dim3;
        double3              origin3;
        double3              dh3;
        std::vector<double>  pressure;
        std::vector<double3> velocity;
        std::vector<double3> force;
    };

    VTIContent content;

    pugi::xml_document doc;

    // 加载 XML 文件
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    CHECK_F(result, "XML 文件解析失败: %s", result.description());

    // 访问根节点
    auto VTKFile   = doc.child("VTKFile");
    auto ImageData = VTKFile.child("ImageData");

    {
        auto vec1 = parseStringToVector<int>(ImageData.attribute("WholeExtent").value());
        auto vec2 = parseStringToVector<double>(ImageData.attribute("Origin").value());
        auto vec3 = parseStringToVector<double>(ImageData.attribute("Spacing").value());

        content.dim3.x = vec1[1] - vec1[0] + 1;
        content.dim3.y = vec1[3] - vec1[2] + 1;
        content.dim3.z = vec1[5] - vec1[4] + 1;

        content.origin3.x = vec2[0];
        content.origin3.y = vec2[1];
        content.origin3.z = vec2[2];

        content.dh3.x = vec3[0];
        content.dh3.y = vec3[1];
        content.dh3.z = vec3[2];
    }

    for (auto DataArray = ImageData.child("Piece").child("PointData").child("DataArray"); DataArray;
         DataArray      = DataArray.next_sibling("DataArray")) {

        std::string name = DataArray.attribute("Name").as_string();
        if (name == "velocity") {
            content.velocity = base64_to_vector<double3>(DataArray.child_value());
        } else if (name == "pressure") {
            content.pressure = base64_to_vector<double>(DataArray.child_value());
        } else if (name == "force") {
            content.force = base64_to_vector<double3>(DataArray.child_value());
        }
    }
    return content;
}



auto read_pvd(const std::string filename) {

    std::vector<double>      timesteps;
    std::vector<std::string> files;

    pugi::xml_document     doc;
    std::filesystem::path  p(filename.c_str());
    pugi::xml_parse_result result = doc.load_file(filename.c_str());

    for (auto DataSet = doc.child("VTKFile").child("Collection").child("DataSet"); DataSet;
         DataSet      = DataSet.next_sibling("DataSet")) {

        timesteps.push_back(DataSet.attribute("timestep").as_double());
        files.push_back(p.parent_path().string() + "/" + DataSet.attribute("file").as_string());
    }


    struct Data {
        std::vector<double>      timesteps;
        std::vector<std::string> files;
    };
    Data data;
    data.timesteps = timesteps;
    data.files     = files;
    return data;
}

}
