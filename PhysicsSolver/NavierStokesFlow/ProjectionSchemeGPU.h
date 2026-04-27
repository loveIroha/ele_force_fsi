/**
 * @file ProjectionSchemeGPU.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-14
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PROJECTION_SCHEME_H__
#define __PROJECTION_SCHEME_H__

#include <io/writeVTK.h>

#include "PressureSolverGPU.h"
#include "TentitiveVelocityGPU.h"

namespace pangu {

template <typename PressureType, typename VelocityType>
class ProjectionSchemeGPU {
  public:
    std::shared_ptr<TentitiveVelocityGPU<PressureType, VelocityType>> tv;
    std::shared_ptr<PressureSolverGPU<PressureType, VelocityType>>    ps;

    bool silent = false;

    double _t;
    double _dt;
    double _nu;
    double _rho;

    int3    _dim;
    double3 _dh;
    double3 _origin;

    VelocityType u_;
    VelocityType u;
    VelocityType f;
    VelocityType b_u;
    VelocityType r_u;

    StdVector<double, double3> u_boundary;

    PressureType p;
    PressureType b_p;
    PressureType r_p;

    PressureBCgpu pbcs;
    VelocityBCgpu vbcs;

  public:
    ProjectionSchemeGPU(int3 dim, double3 dh, double dt, double nu, double rho)
        : _dt(dt), _nu(nu), _dim(dim), _dh(dh), u_(dim), u(dim), f(dim), b_u(dim), r_u(dim), u_boundary(dim), p(dim),
          b_p(dim), r_p(dim), pbcs(dim), vbcs(dim) {
        tv = std::make_shared<TentitiveVelocityGPU<PressureType, VelocityType>>(dim, dh, dt, nu);
        ps = std::make_shared<PressureSolverGPU<PressureType, VelocityType>>(dim, dh, dt);
    }

    void set_dt(double dt) {
        _dt     = dt;
        tv->_dt = dt;
        ps->_dt = dt;
    }

    void set_t(double t) {
        _t     = t;
        tv->_t = t;
        ps->_t = t;
    }

    std::vector<double3> solveOneStep(const std::vector<double3>& fn, std::vector<double3>& un) {
        // TODO : set boundary velocity
        VelocityType ffn(_dim);
        VelocityType uun(_dim);
        ffn.set(fn);
        uun.set(un);
        solveOneStep(ffn, uun);
        std::vector<double3> rres(fn.size());
        u.get(rres);
        return rres;
    }

    VelocityType& solveOneStep(const VelocityType& fn, VelocityType& un) {
        u = un;
        f = fn;
        // 1. solve tentitive velocity u_.
        u_ = un;
        tv->compute_b(b_u, f, u, _t);
        ps->num_presmooth   = 6;
        ps->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            tv->iterate(u_, b_u, 0);
            r_u = 0;
            tv->compute_residuals(u_, b_u, r_u, _dim, _dh);
            if (!silent)
                LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_u.inner(r_u) / r_u.size());
        }

        // 3. solve pressure p.
        p = 0;
        ps->compute_b(b_p, u_);
        ps->num_presmooth   = 6;
        ps->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            ps->iterate(p, b_p, 0);
            ps->compute_residuals(p, b_p, r_p, _dim, _dh);
            if (!silent)
                LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_p.inner(r_p) / r_p.size());
        }
        // 4. correct velocity u.
        correct_velocity(u, u_, p);

        return u;
    }

    void correct_velocity(VelocityType& u, const VelocityType& u_, const PressureType& p) {
        CHECK_F(u.use_gpu() == u_.use_gpu() && u.use_gpu() == p.use_gpu(), "All vectors should be on GPU memory.");
        if (u.use_gpu())
            gpu::navier_stokes_flow::correct_velocity(u.data(), u_.data(), p.data(), _dim, _dh, _dt);
        else
            CHECK_F(false, "CPU version hasn't implemented yet.");
    }

    void record() {
        std::string          filename_v = "stokes_flow_velocities" + std::to_string(_t) + ".vti";
        std::string          filename_p = "stokes_flow_pressure" + std::to_string(_t) + ".vti";
        std::vector<double3> data_u(u.size());
        std::vector<double>  data_p(p.size());
        u.get(data_u);
        p.get(data_p);
        write_vtk(_dim, _origin, _dh, data_u, filename_v);
        write_vtk(_dim, _origin, _dh, data_p, filename_p);
    }

    void record(const std::vector<double3>& forces, const std::vector<double3>& velocities, double t) {
        LOG_F(WARNING, "Recording pressure is not implemented.");
        std::string velocities_file = "stokes_flow_velocities" + std::to_string(t) + ".vti";
        std::string forces_file     = "stokes_flow_forces" + std::to_string(t) + ".vti";
        write_vtk(_dim, _origin, _dh, velocities, velocities_file);
        write_vtk(_dim, _origin, _dh, forces, forces_file);
    }
};

} // namespace pangu
#endif