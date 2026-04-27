/**
 * @file test_write_vtk.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-08
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <catch.hpp>
#include <io/writeVTK.h>

int test_write_vtk() {
    std::string filename = "test_write_vtk.vti";

    std::stringstream sstream;

    int3                 dim    = {2, 2, 2};
    double3              origin = {0.1, 0.0, 0.1};
    double3              dh     = {0.1, 0.5, 0.1};
    std::vector<double3> data(27);

    for (size_t i = 0; i < 27; i++) {
        data[i].x = 0;
        data[i].y = 1;
        data[i].z = 2;
    }
    write_vtk(dim, origin, dh, data, filename);
}