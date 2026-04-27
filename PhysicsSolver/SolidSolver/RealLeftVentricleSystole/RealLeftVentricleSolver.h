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

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLawLV.h"

namespace dolfin {



// ========== 周期性压力表达式==========
class PeriodicPressure : public Expression {
public:
    double t_current;
    double cycle_period;
    
    PeriodicPressure() : t_current(0.0), cycle_period(0.8) {}
    
    void eval(Array<double>& values, const Array<double>& x) const override {
        // 周期性取模
        double t = std::fmod(t_current, cycle_period);
        double p_kPa;
        
        if (t < 0.2) {
            // 舒张期充盈: 线性增长
            p_kPa = 1.067 * t / 0.2;
        } else if (t < 0.5) {
            // 舒张末期: 恒定舒张压
            p_kPa = 1.067;
        } else if (t < 0.65) {
            // 等容收缩期: 指数上升
            double dt = t - 0.5;
            p_kPa = 1.067 + 13.46 * (1.0 - std::exp(-dt*dt / 0.004));
        } else if (t < 0.8) {
            // 射血期: 指数下降
            double dt = 0.8 - t;
            p_kPa = 1.067 + 13.46 * (1.0 - std::exp(-dt*dt / 0.004));
        } else {
            // 安全保护(理论上不会到达)
            p_kPa = 1.067;
        }
        
        // 单位转换: kPa → CGS (g/(cm·s²))
        // 1 kPa = 1000 Pa = 10000 g/(cm·s²)
        values[0] = p_kPa * 10000.0;
    }
};

// ========== 周期性主动收缩力表达式 ==========
class PeriodicTension : public Expression {
public:
    double t_current;
    double cycle_period;
    
    PeriodicTension() : t_current(0.0), cycle_period(0.8) {}
    
    void eval(Array<double>& values, const Array<double>& x) const override {
        double t = std::fmod(t_current, cycle_period);
        double T_kPa;
        
        if (t < 0.5) {
            // 舒张期: 无主动收缩
            T_kPa = 0.0;
        } else if (t < 0.65) {
            // 收缩上升期
            double dt = t - 0.5;
            T_kPa = 84.26 * (1.0 - std::exp(-dt*dt / 0.005));
        } else if (t < 0.8) {
            // 收缩下降期
            double dt = 0.8 - t;
            T_kPa = 84.26 * (1.0 - std::exp(-dt*dt / 0.005));
        } else {
            T_kPa = 0.0;
        }
        
        // 单位转换
        values[0] = T_kPa * 10000.0;
    }
};

class RealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::FunctionSpace, ConstitutiveLawLV::BilinearForm,
                                ConstitutiveLawLV::LinearForm> {
  public:
    // std::array<double, 5> time_array     = {0.0, 0.8, 0.8, 1.5, 4.0};
    // std::array<double, 5> pressure_array = {0.0, 8.0, 8.0, 100.0, 100.0};
    // std::array<double, 5> tension_array  = {0.0, 0.0, 0.0, 100.0, 100.0};
    // LinearInterpolator<5> pressure_interpolator;
    // LinearInterpolator<5> tension_interpolator;

    // std::shared_ptr<SourceScalar> pressure;
    // std::shared_ptr<SourceScalar> tension;



    std::shared_ptr<PeriodicPressure> pressure;
    std::shared_ptr<PeriodicTension> tension;

    double _kappa;
    double _beta;

    RealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<std::size_t>> material_types,
                            std::shared_ptr<MeshFunction<std::size_t>> boundaries,
                            std::shared_ptr<MeshFunction<double>> f00, std::shared_ptr<MeshFunction<double>> f01,
                            std::shared_ptr<MeshFunction<double>> f02, std::shared_ptr<MeshFunction<double>> s00,
                            std::shared_ptr<MeshFunction<double>> s01, std::shared_ptr<MeshFunction<double>> s02,
                            double diastole_pressure, double systole_pressure, double max_tension,
                            double kappa, double beta, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), 
        //   pressure_interpolator(time_array, pressure_array),
        //   tension_interpolator(time_array, tension_array),
        //   pressure(std::make_shared<SourceScalar>()),
        //   tension(std::make_shared<SourceScalar>()), 
          pressure(std::make_shared<PeriodicPressure>()),
          tension(std::make_shared<PeriodicTension>()), 
          _kappa(kappa), _beta(beta) {
        
        // pressure_array[0] = 0;
        // pressure_array[1] = diastole_pressure * ISUnits::mmHg;
        // pressure_array[2] = diastole_pressure * ISUnits::mmHg;
        // pressure_array[3] = systole_pressure * ISUnits::mmHg;
        // pressure_array[4] = systole_pressure * ISUnits::mmHg;

        // tension_array[0] = 0;
        // tension_array[1] = 0;
        // tension_array[2] = 0;
        // tension_array[3] = max_tension * ISUnits::mmHg;
        // tension_array[4] = max_tension * ISUnits::mmHg;

        LOG_F(INFO, "RealLeftVentricleSolver is called!");

        // Initialize
        _material_types = material_types;
        _boundaries     = boundaries;
        L->pressure = pressure;
        L->tension  = tension;

        // Mark the boundaries
        L->ds = _boundaries;
        // L->dx = _material_types;

        // Set the coefficients
        L->kappa = std::make_shared<Constant>(_kappa);
        L->beta  = std::make_shared<Constant>(_beta);

        // Define fiber directions
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // // Update pressure
        // pressure->data = pressure_interpolator(_t);
        // tension->data  = tension_interpolator(_t);
        // LOG_F(WARNING, "Current pressure: %.5e.Current tension: %.5e. Curent time     : %.5e.\n",
        //       pressure->data, tension->data, _t);
        // // Define displacement
        // auto X = std::make_shared<Function>(V);
        // auto G = std::make_shared<Function>(V);

        // // Set displacement and assemble the right hand side vector
        // X->vector()->set_local(vector_X);
        // L->X = X;

        // // Assemble the right hand side vector
        // assemble(b, *L);

        // // Solve Ax=b with linear solver
        // solve(A, *G->vector(), b, "bicgstab", "amg");
        // G->vector()->get_local(vector_G);



         // Update time in periodic expressions
        pressure->t_current = _t;
        tension->t_current  = _t;
        
        // 获取当前压力和收缩力的值（用于日志输出）
        Array<double> p_val(1), t_val(1), dummy_x(3);
        pressure->eval(p_val, dummy_x);
        tension->eval(t_val, dummy_x);
        
        // 转换为mmHg用于可读性
        double p_mmHg = p_val[0] / ISUnits::mmHg;
        double t_mmHg = t_val[0] / ISUnits::mmHg;
        
        LOG_F(WARNING, "Time: %.5f s | Cycle phase: %.5f s | Pressure: %.2f mmHg | Tension: %.2f mmHg",
              _t, std::fmod(_t, 0.8), p_mmHg, t_mmHg);
        
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X) { return -1e20; }
};
} // namespace dolfin
