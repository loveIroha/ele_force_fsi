/**
 * @file RealLeftVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-28
 * @date 2023-11-02 适应新的流体求解器
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#pragma once

#include "ActiveTension.h"

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>

#include "ConstitutiveLawLV.h"

namespace dolfin {
double current_pressure = 0.0;
double current_tension  = 0.0;
double local_time       = 0.0;

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure(double& _t, double& _dt) : t(_t), dt(_dt) {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        double local_t = std::fmod(t, t_period);
        if (local_t < t_load && t < t_period) { values[0] = p_load * local_t / t_load; }
        if (local_t < t_load && t >= t_period) { values[0] = p_load; }
        if (local_t >= t_load && local_t < t_end_diastole) { values[0] = p_load; }
        if (local_t >= t_end_diastole && local_t < t_end_diastole + t_systole) {
            values[0] = p_load + p_load * 17.75 * (local_t - t_end_diastole) / t_systole;
        }
        if (local_t >= t_end_diastole + t_systole) { values[0] = 18.75 * p_load; }
        current_pressure = values[0];
        local_time       = local_t;
    }
    // Current time
    double& t;
    double& dt;
    double  t_period       = 1.5;
    double  t_systole      = 0.2;
    double  t_load         = 0.8;
    double  p_load         = 8.0 * ISUnits::mmHg;
    double  t_end_diastole = 0.8;
    double  end_time       = 1.5;
};

class RealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm> {
  public:
    std::shared_ptr<WallPressure> pressure = nullptr;
    std::shared_ptr<Tension>      tension  = nullptr;

    RealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                            std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                            std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                            std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                            double kappa, double beta, std::string result_path)
        : GenericSolidSolver(mesh, result_path) {
        LOG_F(WARNING, " Real Left Ventricle Solver with electricity is called!");
        _boundaries = boundaries;

        L->kappa = std::make_shared<Constant>(kappa);
        L->beta  = std::make_shared<Constant>(beta);

        // Define pressure
        pressure    = std::make_shared<WallPressure>(_t, _dt);
        L->pressure = pressure;

        // Define fiber directions
        auto f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        auto s0 = std::make_shared<FiberDirections>(s00, s01, s02);
        L->f0   = f0;
        L->s0   = s0;

        // Define active tension && read from cai.dat
        tension = std::make_shared<Tension>(_displacement, _velocity, f0, _t, _dt);
        tension->read_GPB_data();
        L->tension = tension;
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        tension->Ca_i_max = -1e99;
        tension->Ca_i_min = 1e99;
        tension->Ca_b_max = -1e99;
        tension->Ca_b_min = 1e99;
        tension->T_max    = -1e99;
        tension->J_max    = -1e99;
        tension->J_min    = 1e99;

        // Update variables at the current time
        tension->calculate_cai_current(this->_t, this->_dt);
        tension->calculate_lambda();
        tension->calculate_Tactive();
        printf("ca_i_current: %.5e, ca_i_next:   %.5e, ca_i_next_next: %.5e\n", tension->cai_current, tension->cai_next,
               tension->cai_next_next);

        // 检查 tension->t 与 this->_t 是否相同
        // printf("pressure t:   %.5e, tension t:   %.5e, this t:         %.5e\n", pressure->t, tension->t, this->_t);
        // printf("pressure dt:  %.5e, tension dt:  %.5e, this dt:        %.5e\n", pressure->dt, tension->dt,
        // this->_dt);
        LOG_F(WARNING, "Ca_i_max:     %.5e, Ca_i_min:    %.5e, Ca_b_max:       %.5e, Ca_b_min:      %.5e", tension->Ca_i_max,
               tension->Ca_i_min, tension->Ca_b_max, tension->Ca_b_min);
        LOG_F(WARNING, "T_max:        %.5e, J_max:       %.5e, J_min:          %.5e", tension->T_max, tension->J_max,
               tension->J_min);
        LOG_F(WARNING, "Current pressure: %.5e. Current tension: %.5e. Curent time: %.5e. Local time: %.5e.",
              current_pressure, current_tension, pressure->t, local_time);
        printf("Current pressure: %.5e. Current tension: %.5e. Curent time: %.5e. Local time: %.5e.\n",
               current_pressure, current_tension, pressure->t, local_time);
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Mark the boundaries
        L->ds = _boundaries;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X) { return -1e20; }
};
} // namespace dolfin
