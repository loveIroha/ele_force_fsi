/// @date 2023-10-19
/// @file ElerianLagrangianInteraction3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef _ELERIAN_LAGRANGIAN_INTERACTION_3D_H_
#define _ELERIAN_LAGRANGIAN_INTERACTION_3D_H_

#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <dolfin.h>

// local function, avoid redefinition.
namespace {

double3 function_mass(const double3& x) { return make_double3(1.0, 1.0, 1.0); }
} // namespace

template <typename VectorType>
class ElerianLagrangianInteraction {
  private:
    void rhs(const std::vector<double3>& q_values, std::vector<double3>& b) {
        CHECK_F(q_values.size() == _solid_mesh->get_quadrature_rules().size(), "Wrong size.");
        CHECK_F(b.size() == _solid_mesh->num_dofs(), "Wrong size.");
        _solid_mesh->assemble_rhs_with_values_on_quadrature(q_values, b);
    }
    std::shared_ptr<BackgroundMesh2> _fluid_mesh;
    std::shared_ptr<ImmersedMeshP1>    _solid_mesh;

    std::vector<double3> quadrature_forces_vector;
    std::vector<double>  quadrature_forces_scalar;
    std::vector<double3> q_solid_velocities;

    std::vector<double3> mass;
    std::vector<double3> b;

    std::shared_ptr<dolfin::Matrix> dolfin_A;
    dolfin::Vector                  dolfin_x;
    dolfin::Vector                  dolfin_b;

  public:
    std::shared_ptr<BackgroundMesh2> get_fluid_mesh() const { return _fluid_mesh; }
    std::shared_ptr<ImmersedMeshP1>    get_solid_mesh() const { return _solid_mesh; }

    ElerianLagrangianInteraction(std::shared_ptr<BackgroundMesh2> fluid_mesh, std::shared_ptr<ImmersedMeshP1> solid_mesh)
        : _fluid_mesh(fluid_mesh), _solid_mesh(solid_mesh) {
        quadrature_forces_vector.resize(_solid_mesh->get_quadrature_rules().size());
        quadrature_forces_scalar.resize(_solid_mesh->get_quadrature_rules().size());
        q_solid_velocities.resize(_solid_mesh->get_quadrature_rules().size());

        b.resize(_solid_mesh->num_dofs());
        mass.resize(_solid_mesh->num_dofs());

        _solid_mesh->set_function("mass", function_mass);
        auto mass_q_values = _solid_mesh->evaluate_function_on_quadrature_points<double3, double>("mass");
        rhs(mass_q_values, mass);

        std::shared_ptr<dolfin::Matrix> dolfin_A;
        dolfin::Vector                  dolfin_x;
        dolfin::Vector                  dolfin_b;
    }

    void set_dolfin_solver(dolfin::Matrix A) {
        dolfin_A = std::make_shared<dolfin::Matrix>(A);
        dolfin_x.init(_solid_mesh->num_dofs() * 3);
        dolfin_b.init(_solid_mesh->num_dofs() * 3);
    }

    ~ElerianLagrangianInteraction() {}

    void distribute_force(std::vector<double3>& fluid_forces, const std::vector<double3>& solid_forces,
                          const std::vector<double4>& q_rules) {
        CHECK_F(solid_forces.size() == _solid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(fluid_forces.size() == _fluid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(q_rules.size() == _solid_mesh->get_quadrature_rules().size(), "Wrong size.");

        _solid_mesh->evaluate_function_on_quadrature_points<double3, double>(solid_forces, quadrature_forces_vector);
        _solid_mesh->distribute_force(quadrature_forces_vector, q_rules, fluid_forces, _fluid_mesh->get_h3(),
                                      _fluid_mesh->get_dim());
    }

    template <typename TV, typename T>
    void distribute_center(std::vector<TV>& fluid_forces, const std::vector<TV>& solid_forces,
                           const std::vector<double4>& q_rules) {
        CHECK_F(solid_forces.size() == _solid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(fluid_forces.size() == _fluid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(q_rules.size() == _solid_mesh->get_quadrature_rules().size(), "Wrong size.");

        // TODO : 判断 quadrature_forces 是标量还是向量
        // if (sizeof(TV) / sizeof(T) == 1)
        // {
        auto& quadrature_forces = quadrature_forces_scalar;
        _solid_mesh->evaluate_function_on_quadrature_points<TV, T>(solid_forces, quadrature_forces);
        _solid_mesh->distribute_center<TV>(quadrature_forces, q_rules, fluid_forces, _fluid_mesh->get_h3(),
                                           _fluid_mesh->get_dim());
        // }
        // else if (sizeof(TV) / sizeof(T) == 3)
        // {
        //     auto &quadrature_forces = quadrature_forces_vector;
        //     _solid_mesh->evaluate_function_on_quadrature_points<TV,
        //     T>(solid_forces, quadrature_forces);
        //     _solid_mesh->distribute_center<TV>(quadrature_forces, q_rules,
        //     fluid_forces, _fluid_mesh->get_h3(), _fluid_mesh->get_dim());
        // }
        // else
        // {
        //     CHECK_F(false, "Not implemented!");
        // }
    }

    // TODO: parameter for lumping matrix or original matrix
    void interpolate_velocity(std::vector<double3>& solid_velocities, const std::vector<double3>& fluid_velocities,
                              const std::vector<double4>& q_rules) {
        CHECK_F(solid_velocities.size() == _solid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(fluid_velocities.size() == _fluid_mesh->num_dofs(), "Wrong size.");
        CHECK_F(q_rules.size() == _solid_mesh->get_quadrature_rules().size(), "Wrong size.");

        _solid_mesh->interpolate_velocity(q_solid_velocities, q_rules, fluid_velocities, _fluid_mesh->get_h3(),
                                          _fluid_mesh->get_dim());

        rhs(q_solid_velocities, b);

        // // NOTE : Method 1 : with mass lumping matrix
        // for (size_t i = 0; i < mass.size(); i++)
        // {
        //     solid_velocities[i] = b[i] / mass[i];
        // }

        // NOTE : Method 2 : with mass original matrix
        // b => dolfin_b;
        auto b_data = (double*)b.data();
        for (size_t i = 0; i < b.size() * 3; i++)
            dolfin_b.setitem(i, b_data[i]);

        dolfin::solve(*dolfin_A, dolfin_x, dolfin_b, "cg", "amg");

        // dolfin_x => x;
        auto x_data = (double*)solid_velocities.data();
        for (size_t i = 0; i < solid_velocities.size() * 3; i++)
            x_data[i] = dolfin_x[i];
    }
};

#endif