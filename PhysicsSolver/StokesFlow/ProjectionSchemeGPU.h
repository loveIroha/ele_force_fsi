/**
 * @file ProjectionSchemeGPUGPU.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-22
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __PROJECTION_SCHEME_GPU_H__
#define __PROJECTION_SCHEME_GPU_H__

#include <io/writeVTK.h>

#include "PressureSolverGPU.h"
#include "TentitiveVelocityGPU.h"

namespace pangu {

template <typename PressureType, typename VelocityType, typename PressureBoundaryType, typename VelocityBoundaryType>
class ProjectionSchemeGPU {
  private:
    using LocalTentitiveVelocity = TentitiveVelocityGPU<PressureType, VelocityType, VelocityBoundaryType>;
    using LocalPressureSolver    = PressureSolverGPU<PressureType, VelocityType, PressureBoundaryType>;

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
    PressureType s;
    PressureType b_p;
    PressureType r_p;

    PressureBoundaryType pbcs;
    VelocityBoundaryType vbcs;
    VTIWriter            velocity_writer;
    VTIWriter            pressure_writer;
    VTIWriter            force_writer;

  public:
    ProjectionSchemeGPU(int3 dim, double3 dh, double dt, double nu, double rho,
                        std::string velocity_file = "fluid/velocity/data.pvd",
                        std::string pressure_file = "fluid/pressure/data.pvd",
                        std::string force_file    = "fluid/force/data.pvd")
        : _dt(dt), _nu(nu), _dim(dim), _dh(dh), u_(dim), u(dim), f(dim), b_u(dim), r_u(dim), p(dim), s(dim), b_p(dim),
          r_p(dim), pbcs(dim), vbcs(dim), velocity_writer(velocity_file), pressure_writer(pressure_file),
          force_writer(force_file) {
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

    std::vector<double3> solveOneStep(const std::vector<double3>& vector_fn, const std::vector<double3>& vector_un) {
        auto n = vector_fn.size();

        std::vector<double3> vector_u(n);
        std::vector<double>  vector_p(n);

        solveOneStep(vector_fn, vector_un, vector_u, vector_p);

        u.get(vector_u);

        return vector_u;
    }

    void solveOneStep(const std::vector<double3>& vector_fn, const std::vector<double3>& vector_un,
                      std::vector<double3>& vector_u, std::vector<double>& vector_p) {
        auto n = vector_fn.size();

        VelocityType fn(_dim);
        VelocityType un(_dim);

        fn.set(vector_fn);
        un.set(vector_un);

        solveOneStep(fn, un, s, u, p, false);

        vector_u.resize(n);
        vector_p.resize(n);

        u.get(vector_u);
        p.get(vector_p);
    }

    void solveOneStep(const std::vector<double3>& vector_fn, const std::vector<double3>& vector_un,
                      const std::vector<double>& vector_sn, std::vector<double3>& vector_u,
                      std::vector<double>& vector_p) {
        auto n = vector_fn.size();

        VelocityType fn(_dim);
        VelocityType un(_dim);
        PressureType sn(_dim);

        fn.set(vector_fn);
        un.set(vector_un);
        sn.set(vector_sn);

        solveOneStep(fn, un, sn, u, p, true);

        vector_u.resize(n);
        vector_p.resize(n);

        u.get(vector_u);
        p.get(vector_p);
    }

    void solveOneStep(const VelocityType& fn, const VelocityType& un, const PressureType& sn, VelocityType& u,
                      PressureType& p, bool with_source = false) {
        LOG_F(INFO, "The time step for Stokes solver    :  %.16e!", _dt);

        // Solve tentitive velocity u_
        u_ = un;                        // Inital guess for tentitive velocity u_
        tv->compute_b(b_u, fn, un, _t); // Compute the right hand side
        tv->num_presmooth   = 6;
        tv->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            tv->iterate(u_, b_u, 0);
            r_u = 0;
            tv->compute_residuals(u_, b_u, r_u, vbcs, _dim, _dh);
            LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_u.inner(r_u) / r_u.size());
        }

        // Solve pressure p
        // Compute the right hand side
        if (with_source) {
            ps->compute_b(b_p, u_); // Compute the right hand side
        } else {
            ps->compute_b_with_source(b_p, u_, s);
        }

        ps->num_presmooth   = 6;
        ps->num_aftersmooth = 6;
        for (int i = 0; i < 10; i++) {
            ps->iterate(p, b_p, 0);
            r_p = 0;
            ps->compute_residuals(p, b_p, r_p, pbcs, _dim, _dh);
            LOG_F(INFO, "after %d th iteration, the residual is    :  %.16e!", i, r_p.inner(r_p) / r_p.size());
        }

        correct_velocity(u, u_, p, vbcs); // Correct velocity u
        this->u = u;
        this->p = p;
    }

    void correct_velocity(VelocityType& u, const VelocityType& u_, const PressureType& p,
                          const VelocityBoundaryType& vbcs) {
        CHECK_F(u.use_gpu() == u_.use_gpu() && u.use_gpu() == p.use_gpu(), "All vectors should be on GPU memory.");
        gpu::stokesflow::correct_velocity(u.data(), u_.data(), p.data(), vbcs.data(), _dim, _dh, _dt);
    }

    void record(const std::vector<double3>& force, const std::vector<double3>& velocity,
                const std::vector<double>& pressure, double t) {
        LOG_F(INFO, "Recording velocity, pressure and force.");
        velocity_writer.write(_dim, _origin, _dh, velocity, t);
        pressure_writer.write(_dim, _origin, _dh, pressure, t);
        force_writer.write(_dim, _origin, _dh, force, t);
    }
};

} // namespace pangu
#endif