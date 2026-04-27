

#include <iostream>
#include <sstream>
#include <string>

#include <experimental/filesystem>

class VTIWriter {
  public:
    int         counter = 0;
    std::string _path;
    std::string _filename;
    std::string _pvd_filename;

  public:
    VTIWriter(std::string path) {
        // 文件加不存在时创建文件夹
        // NOTE : 换成C++17标准就不需要使用experimental了。
        std::experimental::filesystem::path p(path.c_str());
        std::experimental::filesystem::create_directories(p.parent_path());

        // 处理文件后缀，如果不以pvd结尾，加上。
        std::string extension = ".pvd";
        if (p.extension().string() != extension) p += extension;
        // 如果文件名除了后缀为空，前面加上个data
        if (p.filename().string() == extension) p = p.parent_path().string() + "/data" + p.filename().string();

        std::cout << p.filename().string() << std::endl;
        std::cout << p.parent_path().string() << std::endl;
        std::cout << p.extension().string() << std::endl;
        std::cout << p.string() << std::endl;

        // 取出文件名
        _filename = p.filename().string();
        _path     = p.string();

        // // 初始化 pvd 文件
        // pugi::xml_document xml_doc;
        // pugi::xml_node vtk_node = xml_doc.append_child("VTKFile");
        // vtk_node.append_attribute("type") = "Collection";
        // vtk_node.append_attribute("version") = "0.1";
        // vtk_node.append_child("Collection");
        // xml_doc.save_file(_filename.c_str(), "  ");

        std::cout << vti_name(30) << std::endl;
        std::cout << pvd_name(_path) << std::endl;
    }
    ~VTIWriter() {}

    std::string pvd_name(std::string _filename = "data.pvd") {
        std::string        filestart, ext = ".pvd";
        std::ostringstream newfilename;

        filestart.assign(_filename, 0, _filename.find_last_of("."));

        newfilename << filestart << ext;
        return newfilename.str();
    }

    // input  : vti_name(30)
    // outpur : data000030.vti
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
        //   pugi::xml_document xml_doc;
        //   pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
        //   CHECK_F(result, "XML parsing error when reading from existing file
        //   '%s'", filename);

        // Get Collection node
        //   pugi::xml_node xml_collections =
        //   xml_doc.child("VTKFile").child("Collection");

        // Get filename
        std::experimental::filesystem::path p(filename);
        std::cout << "time: " << time << "\n"
                  << "step: " << step << "\n"
                  << p.filename().string() << std::endl;

        // Append data set
        //   pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
        //   dataset_node.append_attribute("timestep") = time;
        //   dataset_node.append_attribute("part") = "0";
        //   dataset_node.append_attribute("file") = fname_strip.c_str();

        // Save file
        //   xml_doc.save_file(filename.c_str(), "  ");
    }

    void write_vti_file(
        // const int3 &dim,
        // const double3 &origin,
        // const double3 &dh,
        // const std::vector<double3> &data
        std::string filename) {}

    void write(
        // const int3 &dim,
        // const double3 &origin,
        // const double3 &dh,
        // const std::vector<T> &data
        double time) {
        write_pvd_file(counter, time, vti_name(counter, _path));
        write_vti_file(vti_name(counter, _path));
        counter++;
        std::cout << vti_name(counter, _path) << std::endl;
    }
};

int main() {
    std::string a = "abc.d";
    std::string b = "d";

    std::string filestart, extension;

    extension.assign(a, a.find_last_of("."), a.size());

    VTIWriter c("dsfasdfasdfasdfasdf/fluid/.pvd");
    for (size_t i = 0; i < 100; i++) {
        c.write(0.01 * i);
    }
}