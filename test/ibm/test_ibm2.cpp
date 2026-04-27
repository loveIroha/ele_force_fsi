/**
 * @file ibm2.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief new version of
 * @version 0.1
 * @date 2022-02-10
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>
// #include <PhysicsSolver/ImmersedBoundaryMethod/ibm.h>
#include <PhysicsSolver/SolidSolver/SolidSolver.h>
#include <PhysicsSolver/StokesFlow/ProjectionScheme.h>
#include <io/writeVDB.h>

double3 function_velocities(const double3& x) {
    double3 f = x;
    return f;
}

using namespace pangu;

int main() {
    using PressureType = StdVector<double, double>;
    using VelocityType = StdVector<double, double3>;

    // Define fluid mesh and solver
    int          n   = 65;
    double       dt  = 1;
    double       mu  = 0.1;
    double       rho = 1.0;
    int3         dim = make_int3(n, n, n);
    double3      dh  = {1.0 / (dim.x - 1), 1.0 / (dim.y - 1), 1.0 / (dim.z - 1)};
    VelocityType u0(dim);
    VelocityType f(dim);
    std::shared_ptr<ProjectionScheme<PressureType, VelocityType>> projection_scheme;
    auto                                                          fluid_mesh = std::make_shared<BackgroundMesh2>();
    projection_scheme = std::make_shared<ProjectionScheme<PressureType, VelocityType>>(dim, dh, dt, mu, rho);

    // Define solid mesh and solver
    dolfin::Point p2(0.5, 0.4, 0.3);
    dolfin::Point p3(0.9, 0.8, 0.7);
    auto          solid_mesh        = std::make_shared<ImmersedMesh>(std::make_shared<dolfin::Mesh>(
        dolfin::BoxMesh::create({p2, p3}, {8, 8, 8}, dolfin::CellType::Type::tetrahedron)));
    auto          solid_mesh_dolfin = solid_mesh->get_dolfin_mesh();
    auto          _solid_solver     = std::make_shared<dolfin::SolidSolver>(solid_mesh_dolfin, mu);

    // Define interactor
    auto eli = std::make_shared<ElerianLagrangianInteraction<StdVector<double, double3>>>(fluid_mesh, solid_mesh);

    auto& solid_forces     = solid_mesh->set_function("forces", function_velocities);
    auto& solid_velocities = solid_mesh->set_function("velocities", function_velocities);
    auto& solid_positions  = solid_mesh->set_function("positions", function_velocities);

    std::vector<double3> fluid_forces;
    std::vector<double3> fluid_velocities;

    f.get(fluid_forces);
    u0.get(fluid_velocities);

    eli->interpolate_velocity(solid_velocities, fluid_velocities, solid_mesh->get_quadrature_rules());
    eli->distribute_force(fluid_forces, solid_forces, solid_mesh->get_quadrature_rules());

    _solid_solver->record(solid_forces, solid_positions, 0.1);

    int       frame = 1;
    VDBWriter writer;
    writer.addGrid<float, 1>("velocity", fluid_velocities.data(), n, n, n);
    writer.addGrid<float, 3>("force", fluid_forces.data(), n, n, n);
    writer.write("/tmmp/b" + std::to_string(1000 + frame).substr(1) + ".vdb");

    return 0;
}
