/**
 * @file ProjectionScheme.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-01-26
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PROJECTION_SCHEME_H__
#define __PROJECTION_SCHEME_H__

#include <io/writeVTK.h>

#include "PressureSolver.h"
#include "TentitiveVelocity.h"

namespace pangu {

template <typename PressureType, typename VelocityType, typename PressureBoundaryType, typename VelocityBoundaryType>
class ProjectionScheme {
  private:
    using LocalTentitiveVelocity = TentitiveVelocity<PressureType, VelocityType, VelocityBoundaryType>;
    using LocalPressureSolver    = PressureSolver<PressureType, VelocityType, PressureBoundaryType>;

  public:
    std::shared_ptr<LocalTentitiveVelocity> tv;
    std::shared_ptr<LocalPressureSolver>    ps;

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

    PressureBoundaryType pbcs;
    VelocityBoundaryType vbcs;

  public:
    ProjectionScheme(int3 dim, double3 dh, double dt, double nu, double rho)
        : _dt(dt), _nu(nu), _dim(dim), _dh(dh), u_(dim), u(dim), f(dim), b_u(dim), r_u(dim), p(dim), b_p(dim), r_p(dim),
          pbcs(dim), vbcs(dim) {
        tv = std::make_shared<LocalTentitiveVelocity>(dim, dh, dt, nu);
        ps = std::make_shared<LocalPressureSolver>(dim, dh, dt);
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
        VelocityType ffn(_dim);
        VelocityType uun(_dim);
        VelocityType res(_dim);
        ffn.set(fn);
        uun.set(un);
        res = solveOneStep(ffn, uun);
        std::vector<double3> rres(fn.size());
        res.get(rres);
        return rres;
    }

    VelocityType& solveOneStep(const VelocityType& fn, VelocityType& un) {
        u = un;
        f = fn;
        // 1. solve tentitive velocity u_.

        // tv->apply_dirichlet_bcs(u_);
        u_ = un;
        tv->compute_b(b_u, f, u, _t);
        tv->num_presmooth   = 6;
        tv->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            tv->iterate(u_, b_u, 0);
            r_u = 0;
            tv->compute_residuals(u_, b_u, r_u, vbcs, _dim, _dh);
            LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_u.inner(r_u) / r_u.size());
        }

        // 3. solve pressure p.
        ps->compute_b(b_p, u_);
        ps->num_presmooth   = 6;
        ps->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            ps->iterate(p, b_p, 0);
            // r_p = 0;
            // LOG_F(INFO, "%e!", p.sum()/p.size());
            // ps->compute_residuals(p, b_p, r_p, _dim, _dh);
            // LOG_F(INFO, "after %ld th iteration, the residual is    :  %e!", i,
            // r_p.inner(r_p)/r_p.size()); p-=(p.sum()/p.size());
            ps->compute_residuals(p, b_p, r_p, pbcs, _dim, _dh);
            LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_p.inner(r_p) / r_p.size());
            // LOG_F(INFO, "%e!", p.sum()/p.size());
        }
        // 4. correct velocity u.
        correct_velocity(u, u_, p, vbcs);

        return u;
    }

    void correct_velocity(VelocityType& u, const VelocityType& u_, const PressureType& p,
                          const VelocityBoundaryType& vbcs) {
        auto _u    = u.data();
        auto _u_   = u_.data();
        auto _p    = p.data();
        auto _vbcs = vbcs.data();

        auto xstride = 1;
        auto ystride = _dim.x;
        auto zstride = _dim.x * _dim.y;

        for (int i = 0; i < _dim.x; i++) {
            for (int j = 0; j < _dim.y; j++) {
                for (int k = 0; k < _dim.z; k++) {
                    auto centerindex = i * xstride + j * ystride + k * zstride;
                    if (_vbcs[centerindex] == DIRICHLET) continue; // Dirichlet boundary conditions.

                    auto m = _p[centerindex];
                    auto r = i == _dim.x - 1 ? _p[centerindex - xstride] : _p[centerindex + xstride];
                    auto d = j == _dim.y - 1 ? _p[centerindex - ystride] : _p[centerindex + ystride];
                    auto b = k == _dim.z - 1 ? _p[centerindex - zstride] : _p[centerindex + zstride];

                    _u[centerindex].x = _u_[centerindex].x - _dt * (m - r) / _dh.x;
                    _u[centerindex].y = _u_[centerindex].y - _dt * (m - d) / _dh.y;
                    _u[centerindex].z = _u_[centerindex].z - _dt * (m - b) / _dh.z;
                }
            }
        }
    }

    void record() {
        std::string          filename_v = "fluid/stokes_flow_velocities" + std::to_string(_t) + ".vti";
        std::string          filename_p = "fluid/stokes_flow_pressure" + std::to_string(_t) + ".vti";
        std::vector<double3> data_u(u.size());
        std::vector<double>  data_p(p.size());
        u.get(data_u);
        p.get(data_p);
        write_vtk(_dim, _origin, _dh, data_u, filename_v);
        write_vtk(_dim, _origin, _dh, data_p, filename_p);
    }

    void record(const std::vector<double3>& forces, const std::vector<double3>& velocities, double t) {
        LOG_F(WARNING, "Recording pressure is not implemented.");
        std::string velocities_file = "fluid/velocities" + std::to_string(t) + ".vti";
        std::string forces_file     = "fluid/forces" + std::to_string(t) + ".vti";
        write_vtk(_dim, _origin, _dh, velocities, velocities_file);
        write_vtk(_dim, _origin, _dh, forces, forces_file);
    }
};

} // namespace pangu
#endif