/**
 * @file writeVTK.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-08
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <io/ScopeProfiler.h>
#include <io/loguru.hpp>
#include <io/writeVTK.h>

#include <filesystem>

// 尽量不要放在头文件中。
// 存储单个物理量, 支持float, float[1,2,3,4], double, double[1,2,3,4]等等等等。
template <typename T>
void write_vtk(const int3& dim, const double3& origin, const double3& dh, const std::vector<T>& data,
               std::string filename, std::string arrayname) {

    std::stringstream  sstream;
    pugi::xml_document xml_node;

    sstream << 0 << " " << dim.x - 1 << " " << 0 << " " << dim.y - 1 << " " << 0 << " " << dim.z - 1;
    std::string _1 = sstream.str();

    // empty sstream
    sstream.str("");
    sstream << origin.x << " " << origin.y << " " << origin.z;
    std::string _2 = sstream.str();

    sstream.str("");
    sstream << dh.x << " " << dh.y << " " << dh.z;
    std::string _3 = sstream.str();

    // 写入文件头
    pugi::xml_node vtkfile_node                    = xml_node.append_child("VTKFile");
    vtkfile_node.append_attribute("type")          = "ImageData";
    vtkfile_node.append_attribute("version")       = "2.2";
    pugi::xml_node imagedata_node                  = vtkfile_node.append_child("ImageData");
    imagedata_node.append_attribute("WholeExtent") = _1.c_str();
    imagedata_node.append_attribute("Origin")      = _2.c_str();
    imagedata_node.append_attribute("Spacing")     = _3.c_str();
    pugi::xml_node piece_node                      = imagedata_node.append_child("Piece");
    piece_node.append_attribute("Extent")          = _1.c_str();

    // 写入定义在网格点上的数据
    pugi::xml_node pointdata_node = piece_node.append_child("PointData");

    // 写入一个标量
    pointdata_node.append_attribute("Scalars")              = arrayname.c_str();
    pugi::xml_node temperature_node                         = pointdata_node.append_child("DataArray");
    temperature_node.append_attribute("Name")               = arrayname.c_str();
    temperature_node.append_attribute("type")               = "Float64";
    temperature_node.append_attribute("format")             = "binary";
    temperature_node.append_attribute("NumberOfComponents") = (char)(sizeof(T) / sizeof(double));
    std::uint32_t size                                      = data.size() * sizeof(T);
    std::string   header  = base64_encode((const unsigned char*)&size, 1 * sizeof(std::uint32_t));
    std::string   content = base64_encode((const unsigned char*)data.data(), size);
    header.append(content);
    temperature_node.append_child(pugi::node_pcdata).set_value(header.c_str());

    // 文件夹不存在时创建文件夹
    std::filesystem::path p(filename.c_str());
    std::filesystem::create_directories(p.parent_path());

    // 写入定义在网格单元上的数据
    pugi::xml_node celldata_node = piece_node.append_child("CellData");

    // 输出文件，并打印结果
    if (xml_node.save_file(filename.c_str())) {
        LOG_F(INFO, "Saving result successfully!");
    } else {
        LOG_F(WARNING, "Failed to save the result!");
    }
}

// 将速度、力、压强、三个物理量存储到同一个向量中，T 为 double，float等等，TV 为
// double3, float3 等等等等。
template <typename T, typename TV>
void write_vtk(const int3& dim, const double3& origin, const double3& dh, const std::vector<TV>& velocity,
               const std::vector<TV>& force, const std::vector<T>& pressure, std::string filename,
               std::string arrayname) {
    ScopeProfiler _{__func__};

    std::stringstream sstream;

    sstream << 0 << " " << dim.x - 1 << " " << 0 << " " << dim.y - 1 << " " << 0 << " " << dim.z - 1;
    std::string _1 = sstream.str();

    // empty sstream
    sstream.str("");
    sstream << origin.x << " " << origin.y << " " << origin.z;
    std::string _2 = sstream.str();

    sstream.str("");
    sstream << dh.x << " " << dh.y << " " << dh.z;
    std::string _3 = sstream.str();

    pugi::xml_document xml_node;

    // 写入文件头
    pugi::xml_node vtkfile_node                    = xml_node.append_child("VTKFile");
    vtkfile_node.append_attribute("type")          = "ImageData";
    vtkfile_node.append_attribute("version")       = "2.2";
    pugi::xml_node imagedata_node                  = vtkfile_node.append_child("ImageData");
    imagedata_node.append_attribute("WholeExtent") = _1.c_str();
    imagedata_node.append_attribute("Origin")      = _2.c_str();
    imagedata_node.append_attribute("Spacing")     = _3.c_str();
    pugi::xml_node piece_node                      = imagedata_node.append_child("Piece");
    piece_node.append_attribute("Extent")          = _1.c_str();

    // 写入定义在网格点上的数据
    pugi::xml_node pointdata_node = piece_node.append_child("PointData");

    // 写入标量
    pointdata_node.append_attribute("Scalars")              = "pressure";
    pugi::xml_node temperature_node                         = pointdata_node.append_child("DataArray");
    temperature_node.append_attribute("Name")               = "pressure";
    temperature_node.append_attribute("type")               = "Float64";
    temperature_node.append_attribute("format")             = "binary";
    temperature_node.append_attribute("NumberOfComponents") = (char)(sizeof(T) / sizeof(T));
    std::uint32_t size                                      = pressure.size() * sizeof(T);
    std::string   header  = base64_encode((const unsigned char*)&size, 1 * sizeof(std::uint32_t));
    std::string   content = base64_encode((const unsigned char*)pressure.data(), size);
    header.append(content);
    temperature_node.append_child(pugi::node_pcdata).set_value(header.c_str());

    // 写入向量
    //   <PointData Vectors="vector_data">
    //     <DataArray type="Float32" Name="vector_data" NumberOfComponents="2"
    //     format="ascii">
    //       1.0 0.0 0.0
    //       0.0 -1.0 0.0
    //       0.0 1.0 0.0
    //       -1.0 0.0 0.0
    //     </DataArray>
    //   </PointData>
    pointdata_node.append_attribute("Vectors") = "velocity force";

    // 速度
    pugi::xml_node velocity_node                         = pointdata_node.append_child("DataArray");
    velocity_node.append_attribute("Name")               = "velocity";
    velocity_node.append_attribute("type")               = "Float64";
    velocity_node.append_attribute("format")             = "binary";
    velocity_node.append_attribute("NumberOfComponents") = (char)(sizeof(TV) / sizeof(T));
    size                                                 = velocity.size() * sizeof(TV);
    header  = base64_encode((const unsigned char*)&size, 1 * sizeof(std::uint32_t));
    content = base64_encode((const unsigned char*)velocity.data(), size);
    header.append(content);
    velocity_node.append_child(pugi::node_pcdata).set_value(header.c_str());

    // 体力
    pugi::xml_node force_node                         = pointdata_node.append_child("DataArray");
    force_node.append_attribute("Name")               = "force";
    force_node.append_attribute("type")               = "Float64";
    force_node.append_attribute("format")             = "binary";
    force_node.append_attribute("NumberOfComponents") = (char)(sizeof(TV) / sizeof(T));
    size                                              = force.size() * sizeof(TV);
    header  = base64_encode((const unsigned char*)&size, 1 * sizeof(std::uint32_t));
    content = base64_encode((const unsigned char*)force.data(), size);
    header.append(content);
    force_node.append_child(pugi::node_pcdata).set_value(header.c_str());

    // 文件加不存在时创建文件夹
    std::filesystem::path p(filename.c_str());
    std::filesystem::create_directories(p.parent_path());

    // 写入定义在网格单元上的数据
    pugi::xml_node celldata_node = piece_node.append_child("CellData");

    // 输出文件，并打印结果
    if (xml_node.save_file(filename.c_str())) {
        LOG_F(INFO, "Saving result successfully!");
    } else {
        LOG_F(WARNING, "Failed to save the result!");
    }
}

// 重载模板，处理二维单物理量
template <typename T>
void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<T>& data,
               std::string filename, std::string arrayname) {
    int3    dim_3D    = {dim.x, dim.y, 1};
    double3 origin_3D = {origin.x, origin.y, 0};
    double3 dh_3D     = {dh.x, dh.y, 0};
    write_vtk(dim_3D, origin_3D, dh_3D, data, filename, arrayname);
}

// 重载模板，处理二维多物理量数据
void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<double2>& velocity,
               const std::vector<double2>& force, const std::vector<double>& pressure, std::string filename,
               std::string arrayname) {
    std::vector<double3> velocity_3D;
    std::vector<double3> force_3D;
    std::vector<double>  pressure_3D;

    int3    dim_3D    = {dim.x, dim.y, 1};
    double3 origin_3D = {origin.x, origin.y, 0};
    double3 dh_3D     = {dh.x, dh.y, 0};

    for (size_t i = 0; i < velocity.size(); i++) {
        velocity_3D.push_back({velocity[i].x, velocity[i].y, 0});
        force_3D.push_back({force[i].x, force[i].y, 0});
        pressure_3D.push_back(pressure[i]);
    }
    write_vtk(dim_3D, origin_3D, dh_3D, velocity_3D, force_3D, pressure_3D, filename, arrayname);
}

// 重载模板，处理二维多物理量数据
void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<double3>& velocity,
               const std::vector<double3>& force, const std::vector<double>& pressure, std::string filename,
               std::string arrayname) {

    int3    dim_3D    = {dim.x, dim.y, 1};
    double3 origin_3D = {origin.x, origin.y, 0};
    double3 dh_3D     = {dh.x, dh.y, 0};

    write_vtk(dim_3D, origin_3D, dh_3D, velocity, force, pressure, filename, arrayname);
}



// 三维多物理量
template void write_vtk<double, double3>(const int3& dim, const double3& origin, const double3& dh,
                                         const std::vector<double3>& velocity, const std::vector<double3>& force,
                                         const std::vector<double>& pressure, std::string filename,
                                         std::string arrayname);
// 二维单物理量
template void write_vtk<double>(const int2& dim, const double2& origin, const double2& dh,
                                const std::vector<double>& data, std::string filename, std::string arrayname);
template void write_vtk<double2>(const int2& dim, const double2& origin, const double2& dh,
                                 const std::vector<double2>& data, std::string filename, std::string arrayname);
// 三维单物理量
template void write_vtk<double3>(const int3& dim, const double3& origin, const double3& dh,
                                 const std::vector<double3>& data, std::string filename, std::string arrayname);
template void write_vtk<double>(const int3& dim, const double3& origin, const double3& dh,
                                const std::vector<double>& data, std::string filename, std::string arrayname);



bool compressGzipFile(const std::string& sourceFile, const std::string& destFile) {
    std::ifstream inFile(sourceFile, std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open input file: " << sourceFile << std::endl;
        return false;
    }

    gzFile gzOutputFile = gzopen(destFile.c_str(), "wb");
    if (gzOutputFile == NULL) {
        std::cerr << "Failed to open gzipped output file: " << destFile << std::endl;
        return false;
    }

    const int bufferSize = 1280*1024;
    char buffer[bufferSize];
    while (true) {
        inFile.read(buffer, bufferSize);
        int bytesRead = static_cast<int>(inFile.gcount());        
        // printf("inFile.gcount() = %d\n", inFile.gcount());

        if (bytesRead > 0) {
            if (gzwrite(gzOutputFile, buffer, bytesRead) != bytesRead) {
                std::cerr << "Error writing compressed data to file." << std::endl;
                gzclose(gzOutputFile);
                return false;
            }
            // gzflush(gzOutputFile, Z_FINISH); // 刷新缓冲区
        }
        if (inFile.eof()) break;
    }
    inFile.close();
    std::filesystem::remove(sourceFile);
    gzclose(gzOutputFile);
    return true;
}