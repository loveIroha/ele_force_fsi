/**
 * @file test_deformation_gradient.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/SolidSolver/ActiveLeftVentricle/ActiveContraction.h>
#include <PhysicsSolver/SolidSolver/ActiveLeftVentricle/ActiveLeftVentricleSolver.h>
#include <catch.hpp>
#include <loguru/smtp.h>

using UserSolidSolver = dolfin::ActiveLeftVentricleSolver;

using namespace dolfin;

class CurrentConfiguration : public Expression {
  public:
    CurrentConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0] + 2.0 * x[1] + 3.0 * x[2];
        values[1] = 2 * x[1];
        values[2] = 3 * x[2];
    }
};

int test_deformation_gradient() {
    LOG_SCOPE_FUNCTION(WARNING);

    dolfin::XDMFFile mesh_file("/mnt/large2/gjh/realistic_left_ventricle/mesh_scale.xdmf");
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file.read(*solid_mesh_dolfin);
    mesh_file.close();
    auto solid_mesh   = std::make_shared<ImmersedMesh>(solid_mesh_dolfin);
    auto solid_solver = std::make_shared<UserSolidSolver>(solid_mesh_dolfin);

    // define current displacement field
    auto x_current          = std::make_shared<CurrentConfiguration>();
    auto x_current_function = std::make_shared<Function>(solid_solver->V);
    auto u_current_function = std::make_shared<Function>(solid_solver->V);
    x_current_function->interpolate(*x_current);
    dolfin::File("x_current.pvd") << *x_current_function;

    // define initial fiber directions
    auto f00 = std::make_shared<MeshFunction<double>>(
        solid_mesh_dolfin, "/mnt/large2/gjh/realistic_left_ventricle/fibers_0.xml");
    auto f01 = std::make_shared<MeshFunction<double>>(
        solid_mesh_dolfin, "/mnt/large2/gjh/realistic_left_ventricle/fibers_1.xml");
    auto f02 = std::make_shared<MeshFunction<double>>(
        solid_mesh_dolfin, "/mnt/large2/gjh/realistic_left_ventricle/fibers_2.xml");
    auto f0 = std::make_shared<FiberDirections>(f00, f01, f02);

    // define a function of displacement field and fiber directions
    auto deformation_gradient = std::make_shared<DeformationGradient>(x_current_function, u_current_function, f0);
    auto deformation_gradient_function = std::make_shared<dolfin::Function>(solid_solver->V);
    // deformation_gradient_function->interpolate(*deformation_gradient);
    dolfin::File("deformation_gradient.pvd") << *deformation_gradient_function;

    std::vector<double> cai_data;
    std::vector<double> cai_time;
    double              cai_data_max;
    double              cai_time_max;
    ActiveContraction::read_GPB_data(cai_data, cai_time, cai_data_max, cai_time_max);

    double time = 0.3;
    double dt   = 0.01;
    ActiveContraction::cai_current_calculation(cai_data, cai_time, cai_time_max, time, dt);
    double Ca_i;
    double Ca_b;
    double Q1;
    double Q2;
    double Q3;
    double z;
    double lambda     = 10.0;
    double dlambda_dt = 10.0;

    if (time > 0.8)
        deformation_gradient->calculate_T(0.0, 0.0);
    else
        deformation_gradient->set_T(0.0);

    ActiveContraction::NHS_RK2_step(Ca_i, Ca_b, Q1, Q2, Q3, z, lambda, dlambda_dt, time, dt);

    return 1;
}

TEST_CASE("test_deformation_gradient", "[long]") { REQUIRE(test_deformation_gradient()); }