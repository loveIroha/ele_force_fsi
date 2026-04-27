/**
 * @file ImmersedBoundaryMethod.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief Do not use the background mesh defined by dolfin
 * @version 0.2
 * @date 2021-12-12
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _IMMERSED_BOUNDARY_METHOD_H_
#define _IMMERSED_BOUNDARY_METHOD_H_

#include <AlgebraSolver/NonlinearProblem.h>
#include <PhysicsSolver/ImmersedBoundaryMethod/ElerianLagrangianInteraction.h>

// HACK: move it to another place.
//*******************************************************************
#include <algorithm>
template <typename TV>
bool compare_the_norm(const TV& p1, const TV& p2) {
    auto n_p1 = p1.x * p1.x + p1.y * p1.y + p1.z + p1.z;
    auto n_p2 = p2.x * p2.x + p2.y * p2.y + p2.z + p2.z;
    return n_p1 < n_p2;
}

template <typename T>
T find_max(const std::vector<T>& vec) {
    if (vec.empty()) { throw std::invalid_argument("vector is empty"); }
    auto it = std::max_element(vec.begin(), vec.end(), compare_the_norm<T>);
    return *it;
}

//*******************************************************************

template <typename SolidSolver, typename FluidSolver, typename VectorType>
class ImmersedBoundaryMethod : public NonlinearProblem<VectorType> {
    // private:
  public:
    static double3 function_expression(const double3& x) { return {0.0, 0.0, 0.0}; }
    static double3 function_positon(const double3& x) { return x; }

    double _dt;
    double _t   = 0.0;
    int    step = 0;

    std::shared_ptr<BackgroundMesh2> _fluid_mesh;
    std::shared_ptr<ImmersedMesh>    _solid_mesh;
    std::shared_ptr<FluidSolver>     _fluid_solver;
    std::shared_ptr<SolidSolver>     _solid_solver;

    std::vector<double3>& _solid_forces;
    std::vector<double3>& _fluid_forces;
    std::vector<double3>& _solid_velocities;
    std::vector<double3>& _fluid_velocities;
    std::vector<double3>& _solid_positions;
    std::vector<double>&  _fluid_pressures;

    std::vector<double4> _euler_quadrature_rules;

    std::shared_ptr<ElerianLagrangianInteraction<VectorType>> _el_interactor;

  public:
    const std::vector<double3>& get_solid_positions() const { return _solid_positions; }
    void                        get_solid_positions(VectorType& x) const { x.set(_solid_positions); }
    double                      get_dt() const { return _dt; }
    double                      get_t() const { return _t; }

    void set_dt(double dt) {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Set dt as %e", dt);
        _dt = dt;
        _fluid_solver->set_dt(dt);
        _solid_solver->set_dt(dt);
    }

    void set_t(double t) {
        LOG_SCOPE_FUNCTION(INFO);
        _t = t;
        _fluid_solver->set_t(t);
        _solid_solver->set_t(t);
    }

    size_t unkown_size() const { return _solid_positions.size() * 3; }

    ImmersedBoundaryMethod(std::shared_ptr<ImmersedMesh> solid_mesh, std::shared_ptr<BackgroundMesh2> fluid_mesh,
                           std::shared_ptr<SolidSolver> solid_solver, std::shared_ptr<FluidSolver> fluid_solver)
        : _fluid_mesh(fluid_mesh), _solid_mesh(solid_mesh), _fluid_solver(fluid_solver), _solid_solver(solid_solver),
          _solid_forces(solid_mesh->create_function<double3>("forces")),
          _fluid_forces(fluid_mesh->create_function<double3>("forces")),
          _solid_velocities(solid_mesh->create_function<double3>("velocities")),
          _fluid_velocities(fluid_mesh->create_function<double3>("velocities")),
          _solid_positions(solid_mesh->create_function<double3>("positions")),
          _fluid_pressures(fluid_mesh->create_function<double>("pressures")),
          _euler_quadrature_rules(_solid_mesh->get_quadrature_rules().size()),
          _el_interactor(std::make_shared<ElerianLagrangianInteraction<VectorType>>(fluid_mesh, solid_mesh)) {
        LOG_SCOPE_FUNCTION(INFO);
        // 将固体求解器、浸没边界法求解器、流体求解器的当前时间归零
        set_t(0.0);

        _fluid_solver->u0.get(_fluid_velocities);

        // 将固体求解器、浸没边界法求解器、流体求解器的时间步长设为相同
        set_dt(_fluid_solver->get_dt());
        _el_interactor->set_dolfin_solver(_solid_solver->A);

        LOG_F(WARNING, "Immersed Boundary Problem Parameters : dt = %.12e, t = %.12e, nu = %.12e.", _dt, _t,
              _fluid_solver->get_nu());

        _el_interactor->get_fluid_mesh()->list_functions();
        _el_interactor->get_solid_mesh()->list_functions();

        _solid_mesh->set_function("positions", function_positon);
        _solid_mesh->update_Euler_quadrature_rules("positions", _euler_quadrature_rules);
    }

    virtual void Residual(std::shared_ptr<const VectorType> xx, std::shared_ptr<VectorType> rr) override final {
        LOG_SCOPE_FUNCTION(INFO);
        std::vector<double3> x(xx->size());
        std::vector<double3> r(xx->size());
        xx->get(x);

        CHECK_F(xx->size() == rr->size(), "Wrong size.");

        LOG_F(INFO, "Solve solid equations.\n");
        _solid_forces = _solid_solver->solveOneStep_1(x);

        LOG_F(INFO, "Distribute forces from solid to fluid.\n");
        _el_interactor->distribute_force(_fluid_forces, _solid_forces, _euler_quadrature_rules);

        LOG_F(INFO, "Solve NS equations.\n");
        auto temp_fluid_velocities = _fluid_solver->solveOneStep(_fluid_forces, _fluid_velocities);

        LOG_F(INFO, "Interpolate velocity from fluid to solid.\n");
        auto _solid_velocities_new = _solid_velocities;
        _el_interactor->interpolate_velocity(_solid_velocities_new, temp_fluid_velocities, _euler_quadrature_rules);

        LOG_F(INFO, "Calculate the residuals : h(x^*) = x^* - x^n - 0.5 * dt * ( u^* + u^n ).\n");
        _solid_solver->be_scheme_residual(r, x, _solid_positions, _solid_velocities_new, _solid_velocities, _dt);
        rr->set(r);
    }

    void record() {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Record the current fluid and solid status.");
        _fluid_solver->record(_fluid_forces, _fluid_velocities, _fluid_pressures, _t);
        _solid_solver->record(_solid_forces, _solid_positions, _t);
    }

    void advance(const VectorType& xx_star) {
        std::vector<double3> x_star(xx_star.size());
        xx_star.get(x_star);
        advance(x_star);
    }

    // h(x*) = (x*-xn)/dt - u(x*)
    void advance(const std::vector<double3>& x_star) {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Advance one step and update variables.\n\n");

        LOG_F(INFO, "Update solid positions for ImmersedBoundaryMethod and SolidSolver respectively.\n");
        _solid_positions = x_star;

        LOG_F(INFO, "Solve solid equations.\n");
        _solid_forces = _solid_solver->solveOneStep_1(_solid_positions);

        LOG_F(INFO, "Distribute forces from solid to fluid.\n");
        _el_interactor->distribute_force(_fluid_forces, _solid_forces, _euler_quadrature_rules);

        LOG_F(INFO, "Solve NS equations.\n");
        _fluid_velocities = _fluid_solver->solveOneStep(_fluid_forces, _fluid_velocities);

        LOG_F(INFO, "Interpolate velocity from fluid to solid.\n");
        _el_interactor->interpolate_velocity(_solid_velocities, _fluid_velocities, _euler_quadrature_rules);

        LOG_F(INFO, "Update solid positions on quadrature points.\n");
        _solid_mesh->update_Euler_quadrature_rules("positions", _euler_quadrature_rules);

        LOG_F(INFO, "Output the results at time %lf.\n", _t + _dt);
        set_t(_t + _dt);
        step++;
        record();

        LOG_F(INFO, "System Kinetic Energy: %lf", _fluid_mesh->functional("velocities"));
        LOG_F(INFO, "Solid Potential Energy: %lf", _solid_solver->calculate_energy_norm(_solid_positions));
        LOG_F(INFO, "System Total Energy: %lf",
              0.5 * _fluid_mesh->functional("velocities") + _solid_solver->calculate_energy_norm(_solid_positions));

        LOG_F(INFO, "Pressure at the center: %lf", _fluid_pressures[_fluid_mesh->num_dofs() / 2]);
        LOG_F(INFO, "Velocity at the center: %lf, %lf, %lf.", _fluid_velocities[_fluid_mesh->num_dofs() / 2].x,
              _fluid_velocities[_fluid_mesh->num_dofs() / 2].y, _fluid_velocities[_fluid_mesh->num_dofs() / 2].z);
        auto max_displacement = find_max(_solid_positions);
        LOG_F(INFO, "Max solid displacement: %lf, %lf, %lf.", max_displacement.x, max_displacement.y,
              max_displacement.z);
    }

    virtual void explicit_scheme() {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Advance one step with explicit scheme.");

        // Update current time.
        set_t(_t + _dt);

        LOG_F(INFO, "Solve solid equations.");
        _solid_forces = _solid_solver->solveOneStep_1(_solid_positions);

        LOG_F(INFO, "Distribute forces from solid to fluid.");
        _el_interactor->distribute_force(_fluid_forces, _solid_forces, _euler_quadrature_rules);

        LOG_F(INFO, "Solve NS equations.");
        _fluid_solver->solveOneStep(_fluid_forces, _fluid_velocities, _fluid_velocities, _fluid_pressures);
        // _fluid_velocities = _fluid_solver->solveOneStep(_fluid_forces,
        // _fluid_velocities);

        LOG_F(INFO, "Interpolate velocity from fluid to solid.");
        _el_interactor->interpolate_velocity(_solid_velocities, _fluid_velocities, _euler_quadrature_rules);

        LOG_F(INFO, "Update solid positions on quadrature points.");
        _solid_mesh->update_Euler_quadrature_rules("positions", _euler_quadrature_rules);

        // Update solid positions
        for (size_t i = 0; i < _solid_positions.size(); i++) {
            _solid_positions[i].x += _solid_velocities[i].x * _dt;
            _solid_positions[i].y += _solid_velocities[i].y * _dt;
            _solid_positions[i].z += _solid_velocities[i].z * _dt;
        }

        LOG_F(INFO, "System Kinetic Energy: %lf", _fluid_mesh->functional("velocities"));
        LOG_F(INFO, "Solid Potential Energy: %lf", _solid_solver->calculate_energy_norm(_solid_positions));
        LOG_F(INFO, "System Total Energy: %lf",
              0.5 * _fluid_mesh->functional("velocities") + _solid_solver->calculate_energy_norm(_solid_positions));

        LOG_F(INFO, "Pressure at the center: %lf", _fluid_pressures[_fluid_mesh->num_dofs() / 2]);
        LOG_F(INFO, "Velocity at the center: %lf, %lf, %lf.", _fluid_velocities[_fluid_mesh->num_dofs() / 2].x,
              _fluid_velocities[_fluid_mesh->num_dofs() / 2].y, _fluid_velocities[_fluid_mesh->num_dofs() / 2].z);
        auto max_displacement = find_max(_solid_positions);
        LOG_F(INFO, "Max solid displacement: %lf, %lf, %lf.", max_displacement.x, max_displacement.y,
              max_displacement.z);
    }

    void constraint_positions() {
        // Constraint _solid_positions
        _solid_solver->constraint_positions(_solid_positions);
        _solid_mesh->update_Euler_quadrature_rules("positions", _euler_quadrature_rules);
    }

    ~ImmersedBoundaryMethod() {}
};

#endif