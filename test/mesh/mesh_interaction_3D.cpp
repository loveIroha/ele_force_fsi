/// @date 2023-10-19
/// @file mesh_interaction_3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#include <AlgebraSolver/algebra.h>
#include <MeshTools/BackgroundMesh3D.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/DistributionInterpolation3D.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction3D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven3D/DiskDriven.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/MeshInteraction3D.h>
#include <io/ScopeProfiler.h>

#include <tuple>
// const int DIM = 3;

double3 function_any_vector(const double3& x) { return make_double3(x.x, x.y, x.z); }

int main_1() {
    LOG_SCOPE_FUNCTION(INFO);
    LOG_F(INFO, "插值算子");

    int3          dim = {32, 32, 32};
    dolfin::Point p2(0.2, 0.2, 0.2);
    dolfin::Point p3(0.8, 0.8, 0.8);
    auto          solid_mesh = std::make_shared<ImmersedMeshP1>(std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
        {p2, p3}, {(size_t)dim.x, (size_t)dim.y, (size_t)dim.z}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();

    std::array<int, DIM>    dim_bg{100, 100, 100};
    std::array<double, DIM> L{1.0, 1.0, 1.0};
    auto                    domain_mesh = std::make_shared<BackgroundMesh3D<3>>(10,     // Nt
                                                             dim_bg, // Nx Ny
                                                             L,      // Lx Ly
                                                             1.0,    // T
                                                             1.0,    // rho
                                                             0.01    // mu
    );

    MeshInteraction3D mesh_interator_3D(solid_mesh, domain_mesh);

    // 创建定义在拉格朗日网格上的速度场
    std::vector<double3>& solid_velocities = solid_mesh->create_function<double3>("velocities");
    std::vector<double3>& solid_forces     = solid_mesh->create_function<double3>("forces");
    solid_mesh->set_function("forces", function_any_vector);
    auto quadrature_rules = solid_mesh->get_quadrature_rules();

    // 调用插值算子
    // 创建定义在背景网格上的速度场
    auto [eulerian_u, eulerian_v, eulerian_w] = domain_mesh->set_function_on_staggered_grid(function_any_vector);
    mesh_interator_3D.interpolate_velocity(solid_velocities, eulerian_u, eulerian_v, eulerian_w, quadrature_rules);

    // 调用分布算子
    std::vector<double> eulerian_velocity_u(domain_mesh->get_size_u());
    std::vector<double> eulerian_velocity_v(domain_mesh->get_size_v());
    std::vector<double> eulerian_velocity_w(domain_mesh->get_size_w());
    mesh_interator_3D.distribute_force(eulerian_velocity_u, eulerian_velocity_v, eulerian_velocity_w, solid_forces,
                                       quadrature_rules);

    printScopeProfiler();
    return 0;
}

int main(int argc, char* argv[]) {

    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;


    Kokkos::initialize(argc, argv);
    main_1();
    Kokkos::finalize();
    return 0;
}