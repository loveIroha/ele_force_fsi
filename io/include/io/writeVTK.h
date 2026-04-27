/**
 * @file writeVTK.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 输出.vti格式的数据。二维数据本质上是三维数据的特殊化，二维输出函数可以直接调用三维输出函数。
 * @version 0.1
 * @date 2022-04-08
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *          
 */

#pragma once

#include <io/loguru.hpp>
#include <vector_types.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zlib.h>

#include "base64.h"
#include "pugixml.hpp"

template <typename T>
void write_vtk(const int3& dim, const double3& origin, const double3& dh, const std::vector<T>& data,
               std::string filename);
template <typename T>
void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<T>& data,
               std::string filename, std::string arrayname);

template <typename T>
void write_vtk(const int3& dim, const double3& origin, const double3& dh, const std::vector<T>& data,
               std::string filename, std::string arrayname);
template <typename T>
void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<std::vector<T>>& data,
               std::string filename, std::string arrayname);


template <typename T, typename TV>
void write_vtk(const int3& dim, const double3& origin, const double3& dh, const std::vector<TV>& velocity,
               const std::vector<TV>& force, const std::vector<T>& pressure, std::string filename,
               std::string arrayname);

void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<double2>& velocity,
               const std::vector<double2>& force, const std::vector<double>& pressure, std::string filename,
               std::string arrayname);

void write_vtk(const int2& dim, const double2& origin, const double2& dh, const std::vector<double3>& velocity,
               const std::vector<double3>& force, const std::vector<double>& pressure, std::string filename,
               std::string arrayname);

bool compressGzipFile(const std::string& sourceFile, const std::string& destFile);


class VTIWriter {
  public:
    int         counter = 0;
    std::string _path;
    std::string _filename;
    std::string _arrayname;
    std::string _pvd_filename;

  public:
    VTIWriter(std::string path, std::string arrayname = "data") : _arrayname(arrayname) {
        
        // 一些变量
        std::string extension = ".pvd";
        
        // 文件加不存在时创建文件夹
        std::filesystem::path p(path.c_str());
        std::filesystem::create_directories(p.parent_path());

        // 处理文件后缀。 
        if (p.extension().string() != extension) p += extension; // 如果不以pvd结尾，加上".pvd";
        if (p.filename().string() == extension) p = p.parent_path().string() + "/data" + p.filename().string(); // 如果只有后缀，无文件名(.pvd)，加前缀"/data" (/data.pvd)

        // 输出文件信息
        LOG_F(INFO, "Creating directory for VTI files...");
        LOG_F(INFO, "File name: %s", p.filename().string().c_str());
        LOG_F(INFO, "Parent path: %s", p.parent_path().string().c_str());
        LOG_F(INFO, "Extension: %s", p.extension().string().c_str());
        LOG_F(INFO, "Full path: %s", p.string().c_str());

        // 取出文件名
        _path     = p.string();
        _filename = p.filename().string();

        // 初始化 pvd 文件
        pugi::xml_document xml_doc;
        pugi::xml_node     vtk_node          = xml_doc.append_child("VTKFile");
        vtk_node.append_attribute("type")    = "Collection";
        vtk_node.append_attribute("version") = "0.1";
        vtk_node.append_child("Collection");
        xml_doc.save_file(_path.c_str(), "  ");

        // 输出测试结果
        LOG_F(INFO, "VTIWriter created.");
        LOG_F(INFO, "File name for VTI file: %s", vti_name(30).c_str());
        LOG_F(INFO, "File name for PVD file: %s", pvd_name(_path).c_str());
    }
    ~VTIWriter() {}

    // TODO : 此函数逻辑重复，需要删除
    std::string pvd_name(std::string _filename = "data.pvd") {
        
        std::string        ext = ".pvd";
        std::string        filestart;
        std::ostringstream newfilename;
        
        filestart.assign(_filename, 0, _filename.find_last_of("."));

        newfilename << filestart << ext;
        return newfilename.str();
    }

    std::string vti_name(const int counter, std::string _filename = "data.vti") {
        std::string        filestart, ext = ".vti";
        std::ostringstream fileid, newfilename;

        fileid.fill('0');
        fileid.width(6);
        fileid << counter;

        filestart.assign(_filename, 0, _filename.find_last_of("."));

        newfilename << filestart << fileid.str() << ext;
        return newfilename.str();
    }

    void write_pvd_file(std::size_t step, double time, std::string filename) {
        pugi::xml_document     xml_doc;
        pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
        CHECK_F(result, "XML parsing error when reading from existing file '%s'", filename.c_str());

        // Get Collection node
        pugi::xml_node xml_collections = xml_doc.child("VTKFile").child("Collection");

        // Get filename
        std::filesystem::path p(vti_name(step, filename));
        std::cout << "time: " << time << "\n"
                  << "step: " << step << "\n"
                  << p.filename().string() << std::endl;

        // Append data set
        pugi::xml_node dataset_node               = xml_collections.append_child("DataSet");
        dataset_node.append_attribute("timestep") = time;
        dataset_node.append_attribute("part")     = "0";
        dataset_node.append_attribute("file")     = p.filename().string().c_str();

        // Save file
        xml_doc.save_file(filename.c_str(), "  ");
    }

    template <typename T>
    void write(const int3& dim, const double3& origin, const double3& dh, const std::vector<T>& data, double time) {
        write_pvd_file(counter, time, pvd_name(_path));
        write_vtk(dim, origin, dh, data, vti_name(counter, _path));
        counter++;
        std::cout << vti_name(counter, _path) << std::endl;
    }

    template <typename T>
    void write(const int2& dim, const double2& origin, const double2& dh, const std::vector<T>& data, double time) {
        write_pvd_file(counter, time, pvd_name(_path));
        write_vtk(dim, origin, dh, data, vti_name(counter, _path), _arrayname);
        counter++;
        std::cout << vti_name(counter, _path) << std::endl;
    }
    void write(const int2& dim, const double2& origin, const double2& dh, const std::vector<double2>& velocity,
               const std::vector<double2>& force, const std::vector<double>& pressure, double time) {
        write_pvd_file(counter, time, pvd_name(_path));
        write_vtk(dim, origin, dh, velocity, force, pressure, vti_name(counter, _path), _arrayname);
        counter++;
        std::cout << vti_name(counter, _path) << std::endl;
    }
    void write(const int3& dim, const double3& origin, const double3& dh, const std::vector<double3>& velocity,
               const std::vector<double3>& force, const std::vector<double>& pressure, double time) {
        write_pvd_file(counter, time, pvd_name(_path));
        write_vtk(dim, origin, dh, velocity, force, pressure, vti_name(counter, _path), _arrayname);
        // 如果进行压缩，那么原来的数据会被删除
        // compressGzipFile(vti_name(counter, _path), vti_name(counter, _path)+".gz");
        counter++;
    }
};
