#include <iostream>
#include <map>
#include <memory>
#include <vector>

// 为数据格式写一个接口，表示数据可以输入输出。
struct FormatBase {
    virtual void output() = 0;
    virtual void load()   = 0;
    virtual ~FormatBase() = default;
    using Ptr             = std::shared_ptr<FormatBase>;
};

// Different implementations of the data format
struct VTKFormat {
    void output() {}
    void load() {}
};
struct VTIFormat {
    void output() {}
    void load() {}
};
struct VDBFormat {
    void output() {}
    void load() {}
};
struct XDMFFormat {
    void output() {}
    void load() {}
};

namespace format_extra_funcs { // 无法为数据格式类增加成员函数，以重载的形式，外挂追加

template <typename TV>
void load(
    VTKFormat& format, 
    const std::vector<TV>& pressure, 
    const std::vector<TV>& velocity,
    const std::vector<TV>& force) 
{

}

void load(VTKFormat& format) {}
void load(VTIFormat& format) {}
void load(VDBFormat& format) {}
void load(XDMFFormat& format) {}
} // namespace format_extra_funcs

template <class Format>
struct FormatImpl : FormatBase {
    Format format;

    void output() override { format.output(); }

    void load() override {
        format_extra_funcs::load(format); // 此函数为前面定义的重载函数
    }
};

struct FormatFactoryBase {
    virtual FormatBase::Ptr create() = 0;
    virtual ~FormatFactoryBase()     = default;

    using Ptr = std::shared_ptr<FormatFactoryBase>;
};

template <class Format>
struct FormatFactoryImpl : FormatFactoryBase {
    FormatBase::Ptr create() override { return std::make_shared<FormatImpl<Format>>(); }
};

template <class Format>
FormatFactoryBase::Ptr makeFactory() {
    return std::make_shared<FormatFactoryImpl<Format>>();
}

struct ReadWriteData {
    inline static const std::map<std::string, FormatFactoryBase::Ptr> factories = {{"VTK", makeFactory<VTKFormat>()},
                                                                                   {"VTI", makeFactory<VTIFormat>()},
                                                                                   {"VDB", makeFactory<VDBFormat>()},
                                                                                   {"XDMF", makeFactory<XDMFFormat>()}};

    void write_data(std::string type) {
        try {
            printf("type: %s\n", type.c_str());
            format = factories.at(type)->create();
        } catch (std::out_of_range&) {
            std::cout << "no such format " << type.c_str() << " type!\n";
            return;
        }
        format->load();
    }

    void output() {
        if (format) format->output();
    }

    FormatBase::Ptr format;
};

int main() {
    ReadWriteData robot;
    robot.write_data("VTK");
    robot.write_data("XDM");
    robot.output();
    return 0;
}