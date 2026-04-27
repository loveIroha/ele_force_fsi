/// @date 2023-10-22
/// @file fe_solver.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/SolidSolver/DiskDriven3D/DiskDriven.h>
#include <io/ScopeProfiler.h>

double3 function_any_vector(const double3& x) {
    ScopeProfiler _{__func__};
    return make_double3(x.x, 0.0, 0.0);
}

int main() {
    int3          dim = {32, 32, 32};
    dolfin::Point p2(0.2, 0.2, 0.2);
    dolfin::Point p3(0.8, 0.8, 0.8);
    auto          solid_mesh = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
        {p2, p3}, {(size_t)dim.x, (size_t)dim.y, (size_t)dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();
    dolfin::DiskDriven disk_driven(solid_mesh->get_dolfin_mesh(), 0.1);

    // 创建位移变量和力变量
    std::vector<double3>& _solid_forces       = solid_mesh->create_function<double3>("forces");
    std::vector<double3>& _solid_displacement = solid_mesh->create_function<double3>("displacement");

    solid_mesh->set_function("displacement", function_any_vector);

    auto _solid_forces_2       = algebra::flatten<double3, double>(_solid_forces);
    auto _solid_displacement_2 = algebra::flatten<double3, double>(_solid_displacement);

    // 求解力
    disk_driven.solveOneStep(_solid_forces_2, _solid_displacement_2);

    // 保存结果
    _solid_forces = algebra::ripple<double3, double>(_solid_forces_2);
    disk_driven.record<double3, double>(_solid_forces, _solid_displacement, 0.0);

    return 0;
}
