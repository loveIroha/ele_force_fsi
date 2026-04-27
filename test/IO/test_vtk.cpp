/// @date 2023-06-13
/// @file test_vtk.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 测试将数据写入vti文件
///
///

#include <io.h>

#include <MeshTools/StaggeredGrid.h>
#include <AlgebraSolver/algebra.h>

int main_1() {

    // 1. 三维多个物理量
    int3                 dim3{2, 2, 2};
    double3              origin3{0.0, 0.0, 0.0};
    double3              dh3{1.0, 1.0, 1.0};
    std::vector<double3> velocity{
        {1.0, 0.0, 3}, {0.0, -1.0, 3}, {0.0, 1.0, 3}, {0.0, 1.0, 3},
        {1.0, 0.0, 3}, {0.0, -1.0, 3}, {0.0, 1.0, 3}, {0.0, 1.0, 3},
    };
    std::vector<double3> force{
        {11.0, 0.0, 13}, {0.0, -11.0, 13}, {0.0, 11.0, 13}, {0.0, 11.0, 13},
        {11.0, 0.0, 13}, {0.0, -11.0, 13}, {0.0, 11.0, 13}, {0.0, 11.0, 13},
    };
    std::vector<double> pressure{1, 2, 3, 4, 5, 6, 7, 8};
    std::string         filename_many  = "a/3D_2x2x2_many.vti";
    std::string         arrayname_many = "3D_2x2x2_many";
    write_vtk(dim3, origin3, dh3, velocity, force, pressure, filename_many, arrayname_many);

    // 2. 二维多个物理量
    int2                 dim{2, 2};
    double2              origin{0.0, 0.0};
    double2              dh{1.0, 1.0};
    std::vector<double>  pressure_2D{1, 2, 3, 4};
    std::vector<double2> velocity_2D{{1.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 1.0}};

    filename_many  = "a/2D_2x2_many.vti";
    arrayname_many = "2D_2x2_many";
    write_vtk(dim, origin, dh, velocity_2D, velocity_2D, pressure_2D, filename_many, arrayname_many);

    // 3. 二维单个物理量
    std::vector<double> data{1, 2, 3, 4};
    std::string         filename  = "a/2D_2x2_simple.vti";
    std::string         arrayname = "2D_2x2_simple";
    write_vtk(dim, origin, dh, data, filename, arrayname);

    // 4. 二维多个物理量，多层数组(一)
    std::vector<std::vector<double>> data_ij{{1, 2}, {3, 4}};
    std::string                      filename_ij  = "a/2D_2x2_simple_ij.vti";
    std::string                      arrayname_ij = "2D_2x2_simple_ij";
    // write_vtk(dim, origin, dh, data_ij, filename_ij, arrayname_ij);

    // 5. 二维多个物理量，多层数组(二)
    std::vector<std::vector<double2>> data_ij_vector{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    std::string                       filename_ij_vector  = "a/2D_2x2_simple_ij_vector.vti";
    std::string                       arrayname_ij_vector = "2D_2x2_simple_ij_vector";
    // write_vtk(dim, origin, dh, data_ij_vector, filename_ij_vector, arrayname_ij_vector);

    // TODO: 三维单个物理量
    // TODO: 三维单个物理量，多层数组
    return 0;
}

int main_2() {
    auto vti_content = IO::read_vtk("a/3D_2x2x2_many.vti");
    auto dim3        = vti_content.dim3;
    auto origin3     = vti_content.origin3;
    auto dh3         = vti_content.dh3;
    auto pressure    = vti_content.pressure;
    auto velocity    = vti_content.velocity;
    auto force       = vti_content.force;

    std::cout << "dim3: " << dim3.x << " " << dim3.y << " " << dim3.z << std::endl;
    std::cout << "origin3: " << origin3.x << " " << origin3.y << " " << origin3.z << std::endl;
    std::cout << "dh3: " << dh3.x << " " << dh3.y << " " << dh3.z << std::endl;

    for (auto& v : velocity) {
        std::cout << v.x << " " << v.y << " " << v.z << std::endl;
    }
    for (auto& f : force) {
        std::cout << f.x << " " << f.y << " " << f.z << std::endl;
    }
    for (auto& p : pressure) {
        std::cout << p << " " << std::endl;
    }

    return 0;
}



int main() {
    const std::string filename    = "/home/kokkos/ssh/npuheart/build_mv/MV/physical0_64_120000_1.0/"
                                    "discretization4.00e-02_5.00e+06_1.00e+08/dilation_3.00e-01_2.00e-01/fluid/data.pvd";
    auto              pvd_content = IO::read_pvd(filename);

    // 定义一个自定义的 lambda 表达式求和
    auto sum_norm2 = [](const std::vector<double3>& vec) -> double {
        return std::accumulate(vec.begin(), vec.end(), 0, [](double acc, double3 val) {
            return acc + val.x * val.x + val.y * val.y + val.z * val.z;
        });
    };

    for (int i = 0; i < pvd_content.files.size(); i++) {
    
        // {int  i           = 800;
        auto vti_content = IO::read_vtk(pvd_content.files[i].c_str());

        auto slice_velocity
            = algebra::slice(vti_content.velocity, vti_content.dim3.x, vti_content.dim3.y, vti_content.dim3.z, 48);
        auto slice_force = algebra::slice(vti_content.force, vti_content.dim3.x, vti_content.dim3.y, vti_content.dim3.z, 48);
        auto slice_pressure
            = algebra::slice(vti_content.pressure, vti_content.dim3.x, vti_content.dim3.y, vti_content.dim3.z, 48);

        int2    dim{vti_content.dim3.x, vti_content.dim3.y};
        double2 origin{0.0, 0.0};
        double2 dh{vti_content.dh3.x, vti_content.dh3.y};

        std::string filename_many  = "a/3D_2x2x2_many.vti";
        std::string arrayname_many = "3D_2x2x2_many";

        write_vtk(dim, origin, dh, slice_velocity, slice_force, slice_pressure, filename_many, arrayname_many);

        LOG_F(INFO, "dim3:     %d %d %d.", vti_content.dim3.x, vti_content.dim3.y, vti_content.dim3.z);
        LOG_F(INFO, "filename: %s", pvd_content.files[i].c_str());
        LOG_F(INFO, "time:     %f", pvd_content.timesteps[i]);
        LOG_F(INFO, "velocity: %f", sum_norm2(vti_content.velocity));

        StaggeredGrid grid(vti_content.dim3, vti_content.origin3, vti_content.dh3);

        double flowrate = 0.0;
        for (int i = 0; i < vti_content.dim3.x; i++) {
            for (int j = 0; j < vti_content.dim3.y; j++) {
                    auto Nx      = vti_content.dim3.x;
                    auto Ny      = vti_content.dim3.y;
                    auto Nz      = vti_content.dim3.z;
                    auto k = 48;
                    auto x       = grid.x<StaggeredGrid::Offset::z>({i, j, k});
                    // LOG_F(INFO, "x: %f %f %f", x[0], x[1], x[2]);
                    if ((x[0] - 5.0) * (x[0] - 5.0) + (x[1] - 5.0) * (x[1] - 5.0) < 3.0 * 3.0) {
                        flowrate
                            += vti_content.velocity[i + j * Nx + k * Nx * Ny].z * vti_content.dh3.x * vti_content.dh3.y;
                    }
            }
        }
        LOG_F(INFO, "flowrate: %f", flowrate);
    }

    // for(auto& t: timesteps) {
    //     std::cout << t << std::endl;
    // }
    return 0;
}
