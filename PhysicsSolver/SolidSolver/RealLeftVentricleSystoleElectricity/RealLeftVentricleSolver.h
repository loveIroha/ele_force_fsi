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

// #include "ActiveTension.h"

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>

#include "ConstitutiveLawLV.h"
#include <algorithm>
#include <cmath>
#include <vector>

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
        values[0] = 0.0;
        double local_t = std::fmod(t, t_period);
        const double p_peak = 12.625 * p_load;
        const double exp_tau = 0.004;

        if (local_t < t_load && t < t_period) {
            values[0] = p_load * local_t / t_load;
        } else if (local_t < t_load && t >= t_period) {
            values[0] = p_load;
        } else if (local_t >= t_load && local_t < t_end_diastole) {
            values[0] = p_load;
        } else if (local_t >= t_end_diastole && local_t < t_end_diastole + t_systole) {
            double dt_local = local_t - t_end_diastole;
            double rise = 1.0 - std::exp(-dt_local * dt_local / exp_tau);
            values[0] = p_load + p_peak * rise;
        } else if (local_t >= t_end_diastole + t_systole && local_t < t_period) {
            double dt_local = t_period - local_t;
            double fall = 1.0 - std::exp(-dt_local * dt_local / exp_tau);
            values[0] = p_load + p_peak * fall;
        } else {
            values[0] = p_load;
        }

        current_pressure = values[0];
        local_time       = local_t;
    }
    // Current time
    double& t;
    double& dt;
    double  t_period       = 0.9;
    double  t_systole      = 0.15;
    double  t_load         = 0.4;
    double  p_load         = 8.0 * ISUnits::mmHg;
    double  t_end_diastole = 0.5;
    double  end_time       = 0.8;
};

class RealLeftVentricleSolver
    : public GenericSolidSolver<ConstitutiveLawLV::Form_a::TestSpace,
                                ConstitutiveLawLV::Form_a,
                                ConstitutiveLawLV::Form_L> {
  public:
    std::shared_ptr<WallPressure> pressure = nullptr;

    std::shared_ptr<Function> XS_func         = nullptr;
    std::shared_ptr<Function> XW_func         = nullptr;
    std::shared_ptr<Function> Zetas_prev_func = nullptr;
    std::shared_ptr<Function> Zetaw_prev_func = nullptr;
    std::shared_ptr<Function> lmbda_prev_func = nullptr;
    std::shared_ptr<Function> lmbda_proj_func = nullptr;

    std::shared_ptr<Constant> dt_mech_const = nullptr;

    std::shared_ptr<ConstitutiveLawLV::Form_a_proj_lmbda> a_proj_form = nullptr;
    std::shared_ptr<ConstitutiveLawLV::Form_L_proj_lmbda> L_proj_form = nullptr;
    std::shared_ptr<LinearVariationalProblem>             proj_problem = nullptr;
    std::shared_ptr<LinearVariationalSolver>              proj_solver  = nullptr;

    std::vector<std::size_t> imm_to_dolfin_scalar;
    std::vector<std::size_t> imm_to_dolfin_vector;

    void set_dofmap_imm_to_dolfin_scalar(const std::vector<std::size_t>& map) {
        imm_to_dolfin_scalar = map;
    }

    void set_dofmap_imm_to_dolfin_vector(const std::vector<std::size_t>& map) {
        imm_to_dolfin_vector = map;
    }

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

        auto S_space = std::make_shared<ConstitutiveLawLV::CoefficientSpace_XS>(_mesh);

        XS_func         = std::make_shared<Function>(S_space);
        XW_func         = std::make_shared<Function>(S_space);
        Zetas_prev_func = std::make_shared<Function>(S_space);
        Zetaw_prev_func = std::make_shared<Function>(S_space);
        lmbda_prev_func = std::make_shared<Function>(S_space);
        lmbda_proj_func = std::make_shared<Function>(S_space);

        Constant zero(0.0);
        Constant one(1.0);
        XS_func->interpolate(zero);
        XW_func->interpolate(zero);
        Zetas_prev_func->interpolate(zero);
        Zetaw_prev_func->interpolate(zero);
        lmbda_prev_func->interpolate(one);

        dt_mech_const = std::make_shared<Constant>(1.0e-3);

        L->XS         = XS_func;
        L->XW         = XW_func;
        L->Zetas_prev = Zetas_prev_func;
        L->Zetaw_prev = Zetaw_prev_func;
        L->lmbda_prev = lmbda_prev_func;
        L->dt_mech    = dt_mech_const;

        a_proj_form = std::make_shared<ConstitutiveLawLV::Form_a_proj_lmbda>(S_space, S_space);
        L_proj_form = std::make_shared<ConstitutiveLawLV::Form_L_proj_lmbda>(S_space);
        L_proj_form->f0 = f0;

        std::vector<std::shared_ptr<const DirichletBC>> no_bcs;
        proj_problem = std::make_shared<LinearVariationalProblem>(a_proj_form, L_proj_form, lmbda_proj_func, no_bcs);
        proj_solver  = std::make_shared<LinearVariationalSolver>(proj_problem);
        proj_solver->parameters["linear_solver"]  = "iterative";
        proj_solver->parameters["preconditioner"] = "petsc_amg";
    }

    // 由电生理求解器/GPB Land模型提供横桥状态
    void set_land_crossbridge_states(const std::vector<double>& xs_vec, const std::vector<double>& xw_vec) {
        if (xs_vec.size() != xw_vec.size()) {
            throw std::runtime_error("set_land_crossbridge_states: XS/XW size mismatch");
        }
        std::vector<double> xs_local;
        XS_func->vector()->get_local(xs_local);
        if (xs_vec.size() != xs_local.size()) {
            throw std::runtime_error("set_land_crossbridge_states: vector size mismatch with solid dofmap");
        }
        XS_func->vector()->set_local(xs_vec);
        XS_func->vector()->apply("insert");
        XW_func->vector()->set_local(xw_vec);
        XW_func->vector()->apply("insert");
    }

    // 力学步后反馈给电生理Land ODE
    void get_mechanics_feedback(std::vector<double>& lmbda, std::vector<double>& zetas,
                                std::vector<double>& zetaw) const {
        lmbda = lmbda_latest;
        zetas = zetas_latest;
        zetaw = zetaw_latest;
    }

    // 导出当前步 Land 主动收缩力场（与 XS/XW 同标量空间）
    void set_land_active_tension_to_function(std::shared_ptr<Function> Ta_func) const {
        if (!Ta_func) {
            throw std::runtime_error("set_land_active_tension_to_function: Ta_func is null");
        }
        std::vector<double> ta_local;
        if (active_tension_latest.empty()) {
            std::vector<double> ref_local;
            XS_func->vector()->get_local(ref_local);
            ta_local.assign(ref_local.size(), 0.0);
        } else {
            ta_local = active_tension_latest;
        }
        std::vector<double> ta_target;
        Ta_func->vector()->get_local(ta_target);
        if (ta_target.size() != ta_local.size()) {
            throw std::runtime_error("set_land_active_tension_to_function: size mismatch, got " +
                                     std::to_string(ta_target.size()) + ", expected " +
                                     std::to_string(ta_local.size()));
        }
        Ta_func->vector()->set_local(ta_local);
        Ta_func->vector()->apply("insert");
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        const double dt_mech = std::max(this->_dt, 1.0e-9);
        dt_mech_const = std::make_shared<Constant>(dt_mech);
        L->dt_mech = dt_mech_const;

        if (!std::isfinite(dt_mech) || dt_mech <= 0.0) {
            throw std::runtime_error("solveOneStep: invalid dt_mech=" + std::to_string(dt_mech));
        }

        auto check_finite_vec = [](const std::vector<double>& v, const char* name) {
            for (size_t i = 0; i < v.size(); ++i) {
                if (!std::isfinite(v[i])) {
                    throw std::runtime_error(
                        std::string("solveOneStep: non-finite ") + name +
                        " at i=" + std::to_string(i) +
                        ", value=" + std::to_string(v[i]));
                }
            }
        };

        std::vector<double> xs_vec;
        std::vector<double> xw_vec;
        XS_func->vector()->get_local(xs_vec);
        XW_func->vector()->get_local(xw_vec);
        check_finite_vec(xs_vec, "XS");
        check_finite_vec(xw_vec, "XW");

        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        const std::size_t expected_size = X->vector()->size();
        if (imm_to_dolfin_vector.empty()) {
            if (vector_X.size() != expected_size) {
                throw std::runtime_error("solveOneStep: displacement size mismatch, got " +
                                         std::to_string(vector_X.size()) + ", expected " +
                                         std::to_string(expected_size));
            }
        } else {
            if (vector_X.size() != imm_to_dolfin_vector.size()) {
                throw std::runtime_error(
                    "solveOneStep: displacement size mismatch for mapping, got " +
                    std::to_string(vector_X.size()) + ", expected " +
                    std::to_string(imm_to_dolfin_vector.size()));
            }
        }

        for (size_t i = 0; i < vector_X.size(); ++i) {
            if (!std::isfinite(vector_X[i])) {
                throw std::runtime_error(
                    "solveOneStep: non-finite displacement input at i=" + std::to_string(i) +
                    ", value=" + std::to_string(vector_X[i]));
            }
        }
        // Set displacement and assemble the right hand side vector
        if (imm_to_dolfin_vector.empty()) {
            X->vector()->set_local(vector_X);
            X->vector()->apply("insert");
        } else {
            std::vector<double> dolfin_vec(expected_size, 0.0);
            std::vector<unsigned char> used(expected_size, 0);
            for (std::size_t i = 0; i < vector_X.size(); ++i) {
                const std::size_t j = imm_to_dolfin_vector[i];
                if (j >= expected_size) {
                    throw std::runtime_error(
                        "solveOneStep: mapped dof out of range at i=" + std::to_string(i) +
                        ", dof=" + std::to_string(j));
                }
                if (used[j]) {
                    throw std::runtime_error(
                        "solveOneStep: duplicate mapped dof at i=" + std::to_string(i) +
                        ", dof=" + std::to_string(j));
                }
                used[j] = 1;
                dolfin_vec[j] = vector_X[i];
            }
            X->vector()->set_local(dolfin_vec);
            X->vector()->apply("insert");
        }
        L->X = X;
        L_proj_form->X = X;

        if (!_boundaries) {
            throw std::runtime_error("solveOneStep: boundaries not set");
        }

        LOG_F(WARNING, "Current pressure: %.5e. Current tension: %.5e. Curent time: %.5e. Local time: %.5e.",
              current_pressure, current_tension, pressure->t, local_time);

        // Mark the boundaries
        L->ds = _boundaries;

        // Assemble the right hand side vector
        assemble(b, *L);

        std::vector<double> lmbda_prev;
        std::vector<double> zetas_prev;
        std::vector<double> zetaw_prev;
        lmbda_prev_func->vector()->get_local(lmbda_prev);
        Zetas_prev_func->vector()->get_local(zetas_prev);
        Zetaw_prev_func->vector()->get_local(zetaw_prev);
        check_finite_vec(lmbda_prev, "lmbda_prev");
        check_finite_vec(zetas_prev, "zetas_prev");
        check_finite_vec(zetaw_prev, "zetaw_prev");

        std::vector<double> b_local;
        b.get_local(b_local);
        check_finite_vec(b_local, "rhs");

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);

        update_land_history(dt_mech);
    }

    double energy_norm(const std::vector<double>& vector_X) { return -1e20; }

  private:
    std::vector<double> lmbda_latest;
    std::vector<double> zetas_latest;
    std::vector<double> zetaw_latest;
    std::vector<double> active_tension_latest;

    void update_land_history(double dt_mech_s) {
        proj_solver->solve();

        std::vector<double> lmbda_new;
        std::vector<double> lmbda_old;
        std::vector<double> zetas_old;
        std::vector<double> zetaw_old;
        std::vector<double> xs_vec;
        std::vector<double> xw_vec;

        auto check_finite_vec = [](const std::vector<double>& v, const char* name) {
            for (size_t i = 0; i < v.size(); ++i) {
                if (!std::isfinite(v[i])) {
                    throw std::runtime_error(
                        std::string("update_land_history: non-finite ") + name +
                        " at i=" + std::to_string(i) +
                        ", value=" + std::to_string(v[i]));
                }
            }
        };

        lmbda_proj_func->vector()->get_local(lmbda_new);
        lmbda_prev_func->vector()->get_local(lmbda_old);
        Zetas_prev_func->vector()->get_local(zetas_old);
        Zetaw_prev_func->vector()->get_local(zetaw_old);
        XS_func->vector()->get_local(xs_vec);
        XW_func->vector()->get_local(xw_vec);

        check_finite_vec(lmbda_new, "lmbda_new");
        check_finite_vec(lmbda_old, "lmbda_old");
        check_finite_vec(zetas_old, "zetas_old");
        check_finite_vec(zetaw_old, "zetaw_old");
        check_finite_vec(xs_vec, "xs_vec");
        check_finite_vec(xw_vec, "xw_vec");

        zetas_latest.assign(zetas_old.size(), 0.0);
        zetaw_latest.assign(zetaw_old.size(), 0.0);

        const double cs     = 40.14;
        const double cw     = 405.86;
        const double As     = 10.0;
        const double Aw     = 10.0;
        const double exp_cs = std::exp(-cs * dt_mech_s);
        const double exp_cw = std::exp(-cw * dt_mech_s);
        double Tref_land = 120 * ISUnits::kPa;

        for (size_t i = 0; i < zetas_latest.size(); ++i) {
            const double dlmbda     = lmbda_new[i] - lmbda_old[i];
            const double dLambda_dt = dlmbda / std::max(dt_mech_s, 1.0e-9);
            zetas_latest[i]         = zetas_old[i] * exp_cs + (As / cs) * dLambda_dt * (1.0 - exp_cs);
            zetaw_latest[i]         = zetaw_old[i] * exp_cw + (Aw / cw) * dLambda_dt * (1.0 - exp_cw);
        }

        Zetas_prev_func->vector()->set_local(zetas_latest);
        Zetas_prev_func->vector()->apply("insert");
        Zetaw_prev_func->vector()->set_local(zetaw_latest);
        Zetaw_prev_func->vector()->apply("insert");
        lmbda_prev_func->vector()->set_local(lmbda_new);
        lmbda_prev_func->vector()->apply("insert");
        lmbda_latest = lmbda_new;

        active_tension_latest.assign(lmbda_new.size(), 0.0);
        double ta_sum = 0.0;
        for (size_t i = 0; i < lmbda_new.size(); ++i) {
            const double lmbda_c  = std::min(lmbda_new[i], 1.2);
            const double h_prima  = 1.0 + 2.3 * (lmbda_c + std::min(lmbda_c, 0.87) - 1.87);
            const double h_lambda = std::max(0.0, h_prima);
            const double Ta       = h_lambda * (Tref_land / 0.25)
                              * (xs_vec[i] * (zetas_latest[i] + 1.0) + xw_vec[i] * zetaw_latest[i]);
            active_tension_latest[i] = Ta;
            ta_sum += Ta;
        }
        current_tension = lmbda_new.empty() ? 0.0 : ta_sum / static_cast<double>(lmbda_new.size());
    }
};
} // namespace dolfin
