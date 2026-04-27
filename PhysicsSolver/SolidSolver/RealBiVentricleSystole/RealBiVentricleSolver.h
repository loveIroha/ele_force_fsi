/**
 * @file RealBiVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-28
 * @date 2023-11-02 适应新的流体求解器
 * @date 2024-03-12 适应新的IBFE系统
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawLV.h"

namespace dolfin {

class FiberDirections : public Expression {
  public:
    // Create expression with 3 components
    FiberDirections(std::shared_ptr<MeshFunction<double>> _c0, std::shared_ptr<MeshFunction<double>> _c1,
                    std::shared_ptr<MeshFunction<double>> _c2)
        : Expression(3), c0(_c0), c1(_c1), c2(_c2) {}

    // Function for evaluating expression on each cell
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        const uint cell_index = cell.index;
        values[0]             = (*c0)[cell_index];
        values[1]             = (*c1)[cell_index];
        values[2]             = (*c2)[cell_index];
    }

    // The data stored in mesh functions
    std::shared_ptr<dolfin::MeshFunction<double>> c0;
    std::shared_ptr<dolfin::MeshFunction<double>> c1;
    std::shared_ptr<dolfin::MeshFunction<double>> c2;
};
class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure(double _diastole_pressure, double _systole_pressure)
        : t(0), diastole_pressure(_diastole_pressure), systole_pressure(_systole_pressure) {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const { values[0] = current_pressure; }
    void calculate_current_pressure(double t) {
        double values[1]      = {0.0};
        double t_load         = 0.8;
        double t_end_diastole = 0.8;
        double t_cycle        = 1.5;
        double local_t        = std::fmod(t, t_cycle);

        // 第一个周期的前 0.2 秒，压力从 0 到 diastole_pressure
        if (local_t < t_load && t < t_cycle) {
            values[0] = diastole_pressure * local_t / t_load;
        } // 第一个周期的舒张期内壁压力线性增长
        if (local_t < t_load && t >= t_cycle) { values[0] = diastole_pressure; } // 第二个周期的舒张期内壁压力恒定
        if (local_t >= t_load && local_t < t_end_diastole) { values[0] = diastole_pressure; }
        // // 1. 通过sin函数加载压力
        // if (local_t >= t_end_diastole && local_t < t_cycle) {
        //     values[0] = std::sin((local_t - t_end_diastole) / (t_cycle - t_end_diastole) * M_PIf32)
        //                     * (systole_pressure - diastole_pressure)
        //                 + diastole_pressure;
        // }
        // 2. 分段线性函数加载压力
        if (local_t >= t_end_diastole && local_t < 0.5 * (t_cycle + t_end_diastole)) {
            values[0]
                = (local_t - t_end_diastole) / (t_cycle - t_end_diastole) * (systole_pressure - diastole_pressure) * 2.0
                  + diastole_pressure;
        }
        if (local_t >= 0.5 * (t_cycle + t_end_diastole) && local_t < t_cycle) {
            values[0] = (local_t - t_cycle) / (t_cycle - t_end_diastole) * (systole_pressure - diastole_pressure) * 2.0
                        + diastole_pressure;
        }
        current_pressure = values[0];
        local_time       = local_t;
    }

    // Current time
    double t;
    double current_pressure;
    double local_time;
    double diastole_pressure;
    double systole_pressure;
};

class ContractTension : public dolfin::Expression {
  public:
    ContractTension(double _max_tension) : max_tension(_max_tension), t(0) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = current_tension; }

    double calculate_current_tension(double t) {
        double values[1]      = {0.0};
        double t_load         = 0.8;
        double t_end_diastole = 0.8;
        double t_cycle        = 1.5;
        double local_t        = std::fmod(t, t_cycle);

        if (local_t < t_end_diastole) { values[0] = 0.0; }
        // 1. 通过sin函数加载压力
        // if (local_t >= t_end_diastole && local_t < t_cycle) {
        //     values[0] = std::sin((local_t - t_end_diastole) / (t_cycle - t_end_diastole) * M_PIf32) * max_tension;
        // }
        // 2. 分段线性函数加载压力
        if (local_t >= t_end_diastole && local_t < 0.5 * (t_cycle + t_end_diastole)) {
            values[0] = (local_t - t_end_diastole) / (t_cycle - t_end_diastole) * max_tension * 2.0;
        }
        if (local_t >= 0.5 * (t_cycle + t_end_diastole) && local_t < t_cycle) {
            values[0] = (local_t - t_cycle) / (t_cycle - t_end_diastole) * max_tension * 2.0;
        }
        current_tension = values[0];
        local_time      = local_t;
        return values[0];
    }
    double current_tension;
    double local_time;
    double max_tension;
    double t;
};

class RealBiVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm> {
  public:
    std::shared_ptr<WallPressure>    pressure_left;
    std::shared_ptr<WallPressure>    pressure_right;
    std::shared_ptr<ContractTension> tension;
    double                           _kappa;
    double                           _beta;

    RealBiVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                          std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                          std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                          std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                          double kappa, double beta, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path),
          pressure_left(std::make_shared<WallPressure>(8.0 * ISUnits::mmHg, 120.0 * ISUnits::mmHg)),
          pressure_right(std::make_shared<WallPressure>(2.0 * ISUnits::mmHg, 30.0 * ISUnits::mmHg)),
          tension(std::make_shared<ContractTension>(600.0 * ISUnits::mmHg)), _kappa(kappa), _beta(beta) {
        LOG_F(INFO, "RealBiVentricleSolver is called!");
        _boundaries       = boundaries;
        L->pressure_left  = pressure_left;
        L->pressure_right = pressure_right;
        L->tension        = tension;

        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Update pressure
        pressure_left->t  = this->_t;
        pressure_right->t = this->_t;
        tension->t        = this->_t;
        pressure_right->calculate_current_pressure(pressure_right->t);
        pressure_left->calculate_current_pressure(pressure_left->t);
        tension->calculate_current_tension(tension->t);
        LOG_F(WARNING, "Left: Current pressure: %.5e. Curent time     : %.5e. Local time      : %.5e.\n",
              pressure_left->current_pressure, pressure_right->t, pressure_right->local_time);
        LOG_F(WARNING, "Right: Current pressure: %.5e. Curent time     : %.5e. Local time      : %.5e.\n",
              pressure_right->current_pressure, pressure_right->t, pressure_right->local_time);
        LOG_F(WARNING, "Current tension : %.5e. Curent time     : %.5e. Local time      : %.5e.\n",
              tension->current_tension, tension->t, tension->local_time);

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
