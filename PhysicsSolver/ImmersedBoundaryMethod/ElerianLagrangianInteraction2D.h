/**
 * @file ElerianLagrangianInteraction.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief Do not use the background mesh defined by dolfin
 * @version 0.1
 * @date 2021-12-09
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _ELERIAN_LAGRANGIAN_INTERACTION_2D_H_
#define _ELERIAN_LAGRANGIAN_INTERACTION_2D_H_

#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BackgroundMesh2D.h>
#include <MeshTools/BasicMesh2D.h>
#include <dolfin.h>

// // local function, avoid redefinition.
// namespace
// {

//     double2 function_mass(const double3 &x)
//     {
//         return make_double3(1.0, 1.0);
//     }
// }

// template <typename VectorType>
// class ElerianLagrangianInteraction
// {
// private:
//     void rhs(const std::vector<double3> &q_values, std::vector<double3> &b)
//     {
//         CHECK_F(q_values.size() ==
//         _solid_mesh->get_quadrature_rules().size(), "Wrong size.");
//         CHECK_F(b.size() == _solid_mesh->num_dofs(), "Wrong size.");
//         _solid_mesh->assemble_rhs_with_values_on_quadrature(q_values, b);
//     }
//     std::shared_ptr<BackgroundMesh2> _fluid_mesh;
//     std::shared_ptr<ImmersedMesh> _solid_mesh;

//     std::vector<double3> quadrature_forces_vector;
//     std::vector<double> quadrature_forces_scalar;
//     std::vector<double3> q_solid_velocities;

//     std::vector<double3> mass;
//     std::vector<double3> b;

//     std::shared_ptr<dolfin::Matrix> dolfin_A;
//     dolfin::Vector dolfin_x;
//     dolfin::Vector dolfin_b;

// public:
//     std::shared_ptr<BackgroundMesh2> get_fluid_mesh() const { return
//     _fluid_mesh; } std::shared_ptr<ImmersedMesh> get_solid_mesh() const {
//     return _solid_mesh; }

//     ElerianLagrangianInteraction(std::shared_ptr<BackgroundMesh2> fluid_mesh,
//     std::shared_ptr<ImmersedMesh> solid_mesh)
//         : _fluid_mesh(fluid_mesh), _solid_mesh(solid_mesh)
//     {

//         quadrature_forces_vector.resize(_solid_mesh->get_quadrature_rules().size());
//         quadrature_forces_scalar.resize(_solid_mesh->get_quadrature_rules().size());
//         q_solid_velocities.resize(_solid_mesh->get_quadrature_rules().size());

//         b.resize(_solid_mesh->num_dofs());
//         mass.resize(_solid_mesh->num_dofs());

//         _solid_mesh->set_function("mass", function_mass);
//         auto mass_q_values =
//         _solid_mesh->evaluate_function_on_quadrature_points<double3,
//         double>("mass"); rhs(mass_q_values, mass);

//         std::shared_ptr<dolfin::Matrix> dolfin_A;
//         dolfin::Vector dolfin_x;
//         dolfin::Vector dolfin_b;
//     }

//     void set_dolfin_solver(dolfin::Matrix A)
//     {
//         dolfin_A = std::make_shared<dolfin::Matrix>(A);
//         dolfin_x.init(_solid_mesh->num_dofs() * 3);
//         dolfin_b.init(_solid_mesh->num_dofs() * 3);
//     }

//     ~ElerianLagrangianInteraction()
//     {
//     }

//     void distribute_force(
//         std::vector<double3> &fluid_forces,
//         const std::vector<double3> &solid_forces,
//         const std::vector<double4> &q_rules)
//     {

//         CHECK_F(solid_forces.size() == _solid_mesh->num_dofs(), "Wrong
//         size."); CHECK_F(fluid_forces.size() == _fluid_mesh->num_dofs(),
//         "Wrong size."); CHECK_F(q_rules.size() ==
//         _solid_mesh->get_quadrature_rules().size(), "Wrong size.");

//         _solid_mesh->evaluate_function_on_quadrature_points<double3,
//         double>(solid_forces, quadrature_forces_vector);
//         _solid_mesh->distribute_force(quadrature_forces_vector, q_rules,
//         fluid_forces, _fluid_mesh->get_h3(), _fluid_mesh->get_dim());
//     }
// };

#endif