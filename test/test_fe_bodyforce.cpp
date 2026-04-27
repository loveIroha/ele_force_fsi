/**
 * @file test_fe_bodyforce.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-07
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BackgroundMesh.h>

double3 function_positon(const double3& x) { return make_double3(x.x * x.x, x.y * x.y, x.z * x.z); }

int main() {
    int3 dim = {1, 1, 1};

    auto basic_mesh = std::make_shared<BackgroundMesh>(dim);

    auto input = basic_mesh->set_function("positions", function_positon);

    auto result = basic_mesh->evaluate_function_on_quadrature_points("positions");

    auto output = result;

    basic_mesh->calculate_body_force_from_displacement_on_quadrature_points(input, output);
}