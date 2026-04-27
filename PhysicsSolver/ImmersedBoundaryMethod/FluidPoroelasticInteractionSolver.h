/// @date 2023-04-22
/// @file FluidPoroelasticInteractionSolver.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 多孔弹性介质与不可压流体的耦合
///
///

#ifndef _FLUID_POROELASTIC_INTERACTION_SOLVER_H_
#define _FLUID_POROELASTIC_INTERACTION_SOLVER_H_

#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod.h>

template <typename SolidSolver, typename FluidSolver, typename VectorType>
class FluidPoroelasticInteractionSolver : public ImmersedBoundaryMethod<SolidSolver, FluidSolver, VectorType> {
  public:
    /* data */
    std::vector<double>& _fluid_purfusion;
    std::vector<double>& _solid_purfusion;

  public:
    FluidPoroelasticInteractionSolver(std::shared_ptr<ImmersedMesh>    solid_mesh,
                                      std::shared_ptr<BackgroundMesh2> fluid_mesh,
                                      std::shared_ptr<SolidSolver>     solid_solver,
                                      std::shared_ptr<FluidSolver>     fluid_solver)
        : ImmersedBoundaryMethod<SolidSolver, FluidSolver, VectorType>(solid_mesh, fluid_mesh, solid_solver,
                                                                       fluid_solver),
          _fluid_purfusion(fluid_mesh->create_function<double>("purfusion")),
          _solid_purfusion(solid_mesh->create_function<double>("purfusion")) {}
    ~FluidPoroelasticInteractionSolver() {}

    using ImmersedBoundaryMethod<SolidSolver, FluidSolver, VectorType>::_el_interactor;
    using ImmersedBoundaryMethod<SolidSolver, FluidSolver, VectorType>::_euler_quadrature_rules;

    virtual void explicit_scheme() override final {
        // LOG_SCOPE_FUNCTION(INFO);
        // LOG_F(INFO, "Advance one step using an explicit scheme.");

        // Update current time.
        // set_t(_t + _dt);

        // LOG_F(INFO, "Solve solid equations.");
        // _solid_forces = _solid_solver->solveOneStep_1(_solid_positions);
        // _solid_purfusion = _solid_solver->solve_s(_solid_positions);

        // LOG_F(INFO, "Distribute forces from solid to fluid.");
        // _el_interactor->distribute_force(this->_fluid_forces,
        // this->_solid_forces, _euler_quadrature_rules);
        _el_interactor->template distribute_center<double, double>(_fluid_purfusion, _solid_purfusion,
                                                                   _euler_quadrature_rules);

        // LOG_F(INFO, "Solve NS equations.");
        // _fluid_solver->solveOneStep(_fluid_forces, _fluid_velocities,
        // _fluid_purfusion, _fluid_velocities, _fluid_pressures);

        // LOG_F(INFO, "Interpolate velocity from fluid to solid.");
        // _el_interactor->interpolate_velocity(_solid_velocities,
        // _fluid_velocities, _euler_quadrature_rules);

        // LOG_F(INFO, "Update solid positions on quadrature points.");
        // _solid_mesh->update_Euler_quadrature_rules("positions",
        // _euler_quadrature_rules);

        // // Update solid positions
        // for (size_t i = 0; i < _solid_positions.size(); i++)
        // {
        //     _solid_positions[i] = _dt * _solid_velocities[i];
        // }
    }
};

#endif
