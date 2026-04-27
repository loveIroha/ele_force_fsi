/// @date 2023-09-27
/// @file StokesFlow3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __STOKES_FLOW_3D__
#define __STOKES_FLOW_3D__

#include <AlgebraSolver/MultiArray.h>
#include <AlgebraSolver/algebra.h>

#include <io/ScopeProfiler.h>
#include <io/loguru.hpp>
#include <io/vector_io.h>
#include <io/writeVTK.h>

#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityU3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityV3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/Kokkos/VelocityW3D_kokkos.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo.h>
#include <PhysicsSolver/StokesFlow3D/NavierStokesSolution3D.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>
#include <PhysicsSolver/StokesFlow3D/Projection3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityU3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityV3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityW3D.h>

// acceleration
#include <PhysicsSolver/StokesFlow3D/Multigrid/MultigridP.h>

#include <advection.h>

namespace stokes_flow {

template <int DIM, class enable = void>
class StokesFlow;

template <int DIM>
class StokesFlow<DIM, std::enable_if_t<DIM == 3>> {
  public:
    std::shared_ptr<NavierStokesDemo> ns_demo = nullptr;

    int    _Nx, _Ny, _Nz, _Nt;
    double _dx, _dy, _dz, _dt;
    double _Lx, _Ly, _Lz, _Lt;
    double _rho, _mu, _t;

    VTIWriter writer;

    // INTERIOR  0
    // DIRICHLET 1
    // NEUMANN   2
    std::array<int, 6> all_boundary_type   = {1, 1, 1, 1, 1, 1}; // FIXME: 未来应该删除这一项
    std::array<int, 6> all_boundary_type_u = {1, 1, 1, 1, 1, 1};
    std::array<int, 6> all_boundary_type_v = {1, 1, 1, 1, 1, 1};
    std::array<int, 6> all_boundary_type_w = {1, 1, 1, 1, 1, 1};
    std::array<int, 6> all_boundary_type_p = {2, 2, 2, 2, 2, 2};

    int    max_iters = 20000;
    double tolerance = 1e-7; // std::sqrt(std::numeric_limits<double>::epsilon());

    using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
    using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

    MultiArrayDouble eu, ev, ew, ep;
    MultiArrayDouble f1, f2, f3, s;
    MultiArrayDouble un, vn, wn, pn;
    MultiArrayDouble u_, v_, w_, p_;
    MultiArrayDouble uh, vh, wh, ph;
    MultiArrayDouble ru, rv, rw, rp;
    MultiArrayDouble bu, bv, bw, bp;
    MultiArrayDouble ubv, vbv, wbv, pbv;
    MultiArrayInt    ubt, vbt, wbt, pbt;

    std::unique_ptr<mykokkos::VelocityUKokkos3D> kokkos_solver_u = nullptr;
    std::unique_ptr<mykokkos::VelocityVKokkos3D> kokkos_solver_v = nullptr;
    std::unique_ptr<mykokkos::VelocityWKokkos3D> kokkos_solver_w = nullptr;
    std::unique_ptr<mykokkos::PressureKokkos3D>  kokkos_solver_p = nullptr;

    std::unique_ptr<LinearAdvection> advection = nullptr;

  public:
    std::array<int, DIM> get_dim_p() const {
        auto dim_p = std::array<int, DIM>();
        dim_p[0]   = _Nx + 2;
        dim_p[1]   = _Ny + 2;
        dim_p[2]   = _Nz + 2;
        return dim_p;
    }

    std::array<int, DIM> get_dim_u() const {
        CHECK_F(DIM >= 2, "DIM can be 1, 2, 3.");
        auto dim_u = get_dim_p();
        dim_u[0] += 1;
        return dim_u;
    }

    std::array<int, DIM> get_dim_v() const {
        CHECK_F(DIM >= 2, "DIM can be 2,3.");
        auto dim_v = get_dim_p();
        dim_v[1] += 1;
        return dim_v;
    }

    std::array<int, DIM> get_dim_w() const {
        CHECK_F(DIM == 3, "DIM must be 3.");
        auto dim_w = get_dim_p();
        dim_w[2] += 1;
        return dim_w;
    }

    void set_dt(double dt) {
        _dt                 = dt;
        advection->ht       = _dt;
        kokkos_solver_u->dt = _dt;
        kokkos_solver_v->dt = _dt;
        kokkos_solver_w->dt = _dt;
        kokkos_solver_p->dt = _dt;
    }

    void set_t(double t) { _t = t; }

    void set_source(const std::vector<double>& eulerian_force_u, const std::vector<double>& eulerian_force_v,
                    const std::vector<double>& eulerian_force_w) {
        f1 = algebra::ripple(eulerian_force_u, f1.size(), f1[0].size()); // 获取上一时刻力场
        f2 = algebra::ripple(eulerian_force_v, f2.size(), f2[0].size()); // 获取上一时刻力场
        f3 = algebra::ripple(eulerian_force_w, f3.size(), f3[0].size()); // 获取上一时刻力场
    }

    auto get_source() const { return std::make_tuple(std::cref(f1), std::cref(f2), std::cref(f3)); }

    void set_velocity(const std::vector<double>& eulerian_velocity_u, const std::vector<double>& eulerian_velocity_v,
                      const std::vector<double>& eulerian_velocity_w) {
        un = algebra::ripple(eulerian_velocity_u, un.size(), un[0].size()); // 获取上一时刻力场
        vn = algebra::ripple(eulerian_velocity_v, vn.size(), vn[0].size()); // 获取上一时刻力场
        wn = algebra::ripple(eulerian_velocity_w, wn.size(), wn[0].size()); // 获取上一时刻力场
    }

    void set_velocity(const MultiArrayDouble& uu, const MultiArrayDouble& vv, const MultiArrayDouble& ww) {
        un = uu;
        vn = vv;
        wn = ww;
    }

    auto get_velocity() const { return std::make_tuple(std::cref(un), std::cref(vn), std::cref(wn)); }

    auto get_velocity_and_pressure() const {
        return std::make_tuple(std::cref(un), std::cref(vn), std::cref(wn), std::cref(pn));
    }

    StokesFlow(std::shared_ptr<NavierStokesDemo> _ns_demo, std::string _output_file = "fluid/data.pvd")
        : ns_demo(_ns_demo), _Nx{ns_demo->Nx}, _Ny{ns_demo->Ny}, _Nz{ns_demo->Nz}, _Nt(ns_demo->Nt), _dx(ns_demo->dx),
          _dy(ns_demo->dy), _dz(ns_demo->dz), _dt(ns_demo->dt), _Lx{ns_demo->Lx}, _Ly{ns_demo->Ly}, _Lz{ns_demo->Lz},
          _Lt{ns_demo->T}, _rho(ns_demo->rho), _mu(ns_demo->mu), _t(0.0), writer{_output_file} {

        CHECK_F(_Nx * _Ny * _Nz > 0, "Variable dimension makes no sense.");

        mykokkos::print_free_memory<Devices::GPU>();
        advection = std::make_unique<LinearAdvection>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz);

        mykokkos::print_free_memory<Devices::GPU>();
        kokkos_solver_u = std::make_unique<mykokkos::VelocityUKokkos3D>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _mu, _rho,
                                                                        tolerance, max_iters);
        mykokkos::print_free_memory<Devices::GPU>();
        kokkos_solver_v = std::make_unique<mykokkos::VelocityVKokkos3D>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _mu, _rho,
                                                                        tolerance, max_iters);
        mykokkos::print_free_memory<Devices::GPU>();
        kokkos_solver_w = std::make_unique<mykokkos::VelocityWKokkos3D>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _mu, _rho,
                                                                        tolerance, max_iters);
        mykokkos::print_free_memory<Devices::GPU>();
        kokkos_solver_p = std::make_unique<mykokkos::PressureKokkos3D>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _mu, _rho,
                                                                       tolerance, max_iters);
        mykokkos::print_free_memory<Devices::GPU>();

        mykokkos::print_free_memory<Devices::CPU>();
        eu = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        ev = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        ew = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        ep = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        un = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        vn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        wn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        pn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        u_ = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        v_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        w_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        p_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        f1 = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        f2 = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        f3 = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        s  = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        bu = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        bv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        bw = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        bp = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        ru = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        rv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        rw = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        rp = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        ubv = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        vbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        wbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        pbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        ubt = algebra::create_multi_array<3, int>({_Nx + 3, _Ny + 2, _Nz + 2});
        vbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 3, _Nz + 2});
        wbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 2, _Nz + 3});
        pbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 2, _Nz + 2});

        mykokkos::print_free_memory<Devices::CPU>();
        uh = algebra::create_multi_array<3, double>({_Nx + 3, _Ny + 2, _Nz + 2});
        vh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 3, _Nz + 2});
        wh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 3});
        ph = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});
        mykokkos::print_free_memory<Devices::CPU>();

        LOG_F(INFO, "rho : %.4f, mu : %.4f, max_iters : %05d.", _rho, _mu, max_iters);
        LOG_F(INFO, "Lx  : %.4f, Ly : %.4f, Lz : %.4f, T  : %.4f.", _Lx, _Ly, _Lz, _Lt);
        LOG_F(INFO, "Nx  : %06d, Ny : %06d, Nz : %06d, Nt : %08d.", _Nx, _Ny, _Nz, _Nt);
        LOG_F(INFO, "dx  : %.4f, dy : %.4f, dz : %.4f, dt : %.8f.", _dx, _dy, _dz, _dt);
    }

    virtual void initial_values() {
        ns_demo->get_array_u(un, 0.0);
        ns_demo->get_array_v(vn, 0.0);
        ns_demo->get_array_w(wn, 0.0);
        ns_demo->get_array_p(pn, 0.0);
    }

    virtual void set_source_terms() {
        ns_demo->get_array_f1(f1, _t);
        ns_demo->get_array_f2(f2, _t);
        ns_demo->get_array_f3(f3, _t);
    }

    void initial_boundary_conditions() {
        // FIXME: individual boundary conditions for every component of velocity.
        ns_demo->get_boundary_type_u(ubt, all_boundary_type);
        ns_demo->get_boundary_type_v(vbt, all_boundary_type);
        ns_demo->get_boundary_type_w(wbt, all_boundary_type);
        ns_demo->get_boundary_type_p(pbt, all_boundary_type_p);
    }

    void change_boundary_types(const std::array<int, 6>& bc) {
        all_boundary_type      = bc;
        all_boundary_type_u    = bc;
        all_boundary_type_v    = bc;
        all_boundary_type_w    = bc;
        all_boundary_type_p[0] = 3 - all_boundary_type_u[0];
        all_boundary_type_p[1] = 3 - all_boundary_type_u[1];
        all_boundary_type_p[2] = 3 - all_boundary_type_v[2];
        all_boundary_type_p[3] = 3 - all_boundary_type_v[3];
        all_boundary_type_p[4] = 3 - all_boundary_type_w[4];
        all_boundary_type_p[5] = 3 - all_boundary_type_w[5];
        initial_boundary_conditions();
        set_boundary_conditions();
    }

    virtual void set_boundary_conditions() {
        ns_demo->get_boundary_values_u(ubv, ubt, _t);
        ns_demo->get_boundary_values_v(vbv, vbt, _t);
        ns_demo->get_boundary_values_w(wbv, wbt, _t);
        ns_demo->get_boundary_values_p(pbv, pbt, _t);
    }

    virtual void post_process(int i) {}

    int benchmark_heat_equations() {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "benchmark heat equations.");

        double final_errors[4] = {};
        // NOTE: 当压强的所有边界条件都是 Neumann 类型时，需要进行额外标记，以便求解器进行额外处理。
        {
            auto all_neumann = all_boundary_type_p[0] == NEUMANN && all_boundary_type_p[1] == NEUMANN
                               && all_boundary_type_p[2] == NEUMANN && all_boundary_type_p[3] == NEUMANN
                               && all_boundary_type_p[4] == NEUMANN && all_boundary_type_p[5] == NEUMANN;
            if (all_neumann) {
                kokkos_solver_p->PureNeumann = true;
                LOG_F(WARNING, "pure neumann boundary conditions are applied!");
            }
        }
        _t = 0.0;
        initial_boundary_conditions();
        initial_values();

        for (int i = 1; i <= _Nt; i++) {
            _t = i * _dt;
            LOG_SCOPE_F(INFO, "Solving heat equations at t = %f", _t);

            set_boundary_conditions();
            set_source_terms();
            ns_demo->make_tentitive_ub(bu, un, f1);
            ns_demo->make_tentitive_vb(bv, vn, f2);
            ns_demo->make_tentitive_wb(bw, wn, f3);
            ns_demo->get_array_bp(bp, _t);

            int3    N  = {_Nx, _Ny, _Nz};
            double3 L  = {_Lx, _Ly, _Lz};
            double3 dh = {_Lx / _Nx, _Ly / _Ny, _Lz / _Nz};

            mykokkos::solve(kokkos_solver_u, ru, uh, ubt, ubv, bu, N, L, _mu, _rho, _dt, max_iters, tolerance);
            mykokkos::solve(kokkos_solver_v, rv, vh, vbt, vbv, bv, N, L, _mu, _rho, _dt, max_iters, tolerance);
            mykokkos::solve(kokkos_solver_w, rw, wh, wbt, wbv, bw, N, L, _mu, _rho, _dt, max_iters, tolerance);
            mykokkos::solve(kokkos_solver_p, rp, ph, pbt, pbv, bp, N, L, _mu, _rho, _dt, max_iters, tolerance);

            pn = ph;
            un = uh;
            vn = vh;
            wn = wh;

            // 计算误差
            ns_demo->get_array_u(u_, _t);
            ns_demo->get_array_v(v_, _t);
            ns_demo->get_array_w(w_, _t);
            ns_demo->get_array_p(p_, _t);
            algebra::axpy(-1.0, u_, un, eu);
            algebra::axpy(-1.0, v_, vn, ev);
            algebra::axpy(-1.0, w_, wn, ew);
            algebra::axpy(-1.0, p_, pn, ep);
            algebra::zero_boundary(eu);
            algebra::zero_boundary(ev);
            algebra::zero_boundary(ew);
            algebra::zero_boundary(ep);
            double sum_eh_squared_u = algebra::norm(eu) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_v = algebra::norm(ev) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_w = algebra::norm(ew) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_p = algebra::norm(ep) * std::sqrt(dh.x * dh.y * dh.z);
            LOG_F(INFO, "sum_eh_squared_u = %.20e", sum_eh_squared_u);
            LOG_F(INFO, "sum_eh_squared_v = %.20e", sum_eh_squared_v);
            LOG_F(INFO, "sum_eh_squared_w = %.20e", sum_eh_squared_w);
            LOG_F(INFO, "sum_eh_squared_p = %.20e", sum_eh_squared_p);

            final_errors[0] = sum_eh_squared_u;
            final_errors[1] = sum_eh_squared_v;
            final_errors[2] = sum_eh_squared_w;
            final_errors[3] = sum_eh_squared_p;

            post_process(i);
        }

        LOG_F(WARNING, "输入参数 : %d %d %d %d", _Nx, _Ny, _Nz, _Nt);
        LOG_F(WARNING, "时空步长 : %.6e %.6e %.6e %.6e", _dx, _dy, _dz, _dt);
        LOG_F(WARNING, "误差统计 : %.6e %.6e %.6e %.6e", final_errors[0], final_errors[1], final_errors[2],
              final_errors[3]);

        return 0;
    }

    int benchmark_stokes_equations() {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "Solving an unsteady Stokes problem...");

        double final_errors[4] = {};
        // NOTE: 当压强的所有边界条件都是 Neumann 类型时，需要进行额外标记，以便求解器进行额外处理。
        {
            auto all_neumann = all_boundary_type_p[0] == NEUMANN && all_boundary_type_p[1] == NEUMANN
                               && all_boundary_type_p[2] == NEUMANN && all_boundary_type_p[3] == NEUMANN
                               && all_boundary_type_p[4] == NEUMANN && all_boundary_type_p[5] == NEUMANN;
            if (all_neumann) {
                kokkos_solver_p->PureNeumann = true;
                LOG_F(WARNING, "pure neumann boundary conditions are applied!");
            }
        }
        _t = 0.0;
        initial_boundary_conditions();
        initial_values();

        for (int i = 1; i <= _Nt; i++) {
            _t = i * _dt;
            LOG_SCOPE_F(INFO, "Solving problem at t = %f", _t);

            // 对流项 ()
            v_ = vn;
            u_ = un;
            w_ = wn;
            set_boundary_conditions();
            set_source_terms();
            ns_demo->make_tentitive_ub(bu, un, f1);
            ns_demo->make_tentitive_vb(bv, vn, f2);
            ns_demo->make_tentitive_wb(bw, wn, f3);

            int3    N  = {_Nx, _Ny, _Nz};
            double3 L  = {_Lx, _Ly, _Lz};
            double3 dh = {_Lx / _Nx, _Ly / _Ny, _Lz / _Nz};

            mykokkos::solve(kokkos_solver_u, ru, u_, ubt, ubv, bu, N, L, _mu, _rho, _dt, max_iters, tolerance);
            mykokkos::solve(kokkos_solver_v, rv, v_, vbt, vbv, bv, N, L, _mu, _rho, _dt, max_iters, tolerance);
            mykokkos::solve(kokkos_solver_w, rw, w_, wbt, wbv, bw, N, L, _mu, _rho, _dt, max_iters, tolerance);

            // uh vh wh
            vh = v_;
            uh = u_;
            wh = w_;

            projection_3D::make_pb(bp, u_, v_, w_, _Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _rho, _mu);

            mykokkos::solve(kokkos_solver_p, rp, ph, pbt, pbv, bp, N, L, _mu, _rho, _dt, max_iters, tolerance);
            projection_3D::correct_velocity(uh, vh, wh, u_, v_, w_, ph, _Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _rho);
            pn = ph;
            un = uh;
            vn = vh;
            wn = wh;

            auto div_result = projection_3D::calculate_velocity_divergence(un, vn, wn, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz);

            LOG_F(INFO, "divergence of the velocity : %.20e", div_result);
            // 计算误差
            ns_demo->get_array_u(u_, _t);
            ns_demo->get_array_v(v_, _t);
            ns_demo->get_array_w(w_, _t);
            ns_demo->get_array_p(p_, _t);
            algebra::axpy(-1.0, u_, un, eu);
            algebra::axpy(-1.0, v_, vn, ev);
            algebra::axpy(-1.0, w_, wn, ew);
            algebra::axpy(-1.0, p_, pn, ep);
            algebra::zero_boundary(eu);
            algebra::zero_boundary(ev);
            algebra::zero_boundary(ew);
            algebra::zero_boundary(ep);
            double sum_eh_squared_u = algebra::norm(eu) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_v = algebra::norm(ev) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_w = algebra::norm(ew) * std::sqrt(dh.x * dh.y * dh.z);
            double sum_eh_squared_p = algebra::norm(ep) * std::sqrt(dh.x * dh.y * dh.z);

            LOG_F(INFO, "sum_eh_squared_u = %.20e", sum_eh_squared_u);
            LOG_F(INFO, "sum_eh_squared_v = %.20e", sum_eh_squared_v);
            LOG_F(INFO, "sum_eh_squared_w = %.20e", sum_eh_squared_w);
            LOG_F(INFO, "sum_eh_squared_p = %.20e", sum_eh_squared_p);

            final_errors[0] = sum_eh_squared_u;
            final_errors[1] = sum_eh_squared_v;
            final_errors[2] = sum_eh_squared_w;
            final_errors[3] = sum_eh_squared_p;

            post_process(i);
        }

        LOG_F(WARNING, "输入参数 : %d %d %d %d", _Nx, _Ny, _Nz, _Nt);
        LOG_F(WARNING, "时空步长 : %.6e %.6e %.6e %.6e", _dx, _dy, _dz, _dt);
        LOG_F(WARNING, "误差统计 : %.6e %.6e %.6e %.6e", final_errors[0], final_errors[1], final_errors[2],
              final_errors[3]);

        return 0;
    }

    void solve_one_step(MultiArrayDouble& uh, MultiArrayDouble& vh, MultiArrayDouble& wh, MultiArrayDouble& ph,
                        MultiArrayDouble& u_, MultiArrayDouble& v_, MultiArrayDouble& w_, MultiArrayDouble& p_,
                        const MultiArrayDouble& f1, const MultiArrayDouble& f2, const MultiArrayDouble& f3,
                        const MultiArrayDouble& s, const MultiArrayDouble& un, const MultiArrayDouble& vn,
                        const MultiArrayDouble& wn, const MultiArrayDouble& pn, const MultiArrayDouble& ubv,
                        const MultiArrayDouble& vbv, const MultiArrayDouble& wbv, const MultiArrayDouble& pbv,
                        const MultiArrayInt& ubt, const MultiArrayInt& vbt, const MultiArrayInt& wbt,
                        const MultiArrayInt& pbt, MultiArrayDouble& bu, MultiArrayDouble& bv, MultiArrayDouble& bw,
                        MultiArrayDouble& bp, MultiArrayDouble& ru, MultiArrayDouble& rv, MultiArrayDouble& rw,
                        MultiArrayDouble& rp, int _Nx, int _Ny, int _Nz, double _dt, double _Lx, double _Ly, double _Lz,
                        double _rho, double _mu, int max_iters, double tolerance) {
        ScopeProfiler _{"kokkos::StokesFlow::solve_one_step::full"};

        // 处理对流项
        v_ = vn;
        u_ = un;
        w_ = wn;

        // ERROR: 此处有可能存在内存泄漏，详见 https://github.com/npuheart/npuheart/issues/46
        // if (USE_ADVECTION) {
        //     LOG_F(WARNING, "计算对流项");
        // }
        if (USE_ADVECTION) {
            // ScopeProfiler _{"kokkos::StokesFlow::solve_one_step::advection"};
            advection->advection(u_, v_, w_);
        }

        ns_demo->make_tentitive_ub(bu, u_, f1);
        ns_demo->make_tentitive_vb(bv, v_, f2);
        ns_demo->make_tentitive_wb(bw, w_, f3);

        // 1. Tentitive velocity
        int3    N  = {_Nx, _Ny, _Nz};
        double3 L  = {_Lx, _Ly, _Lz};
        double3 dh = {_Lx / _Nx, _Ly / _Ny, _Lz / _Nz};

        mykokkos::solve(kokkos_solver_u, ru, u_, ubt, ubv, bu, N, L, _mu, _rho, _dt, max_iters, tolerance);
        mykokkos::solve(kokkos_solver_v, rv, v_, vbt, vbv, bv, N, L, _mu, _rho, _dt, max_iters, tolerance);
        mykokkos::solve(kokkos_solver_w, rw, w_, wbt, wbv, bw, N, L, _mu, _rho, _dt, max_iters, tolerance);

        // NOTE: (u_, v_, w_) are tentitive velocity, which satisfy the boundary conditions.
        // NOTE: Boundary values of (uh, vh, wh) are avoid to be modified when they are updated.
        vh = v_;
        uh = u_;
        wh = w_;

        // 2. Pressure Update
        // TODO: 考虑散度不为零的情况
        projection_3D::make_pb(bp, u_, v_, w_, _Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _rho, _mu);
        mykokkos::solve(kokkos_solver_p, rp, ph, pbt, pbv, bp, N, L, _mu, _rho, _dt, max_iters, tolerance);

        // 3. Velocity Update
        projection_3D::correct_velocity(uh, vh, wh, u_, v_, w_, ph, _Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _rho);
        LOG_F(INFO, "Fluid solver at t = %f, uh_norm2 = %.16e", _t, algebra::norm(uh));
    }

    void record() { record(un, vn, wn, f1, f2, f3, pn, _t); }

    void record(const MultiArrayDouble& u_out, const MultiArrayDouble& v_out, const MultiArrayDouble& w_out,
                const MultiArrayDouble& fu_out, const MultiArrayDouble& fv_out, const MultiArrayDouble& fw_out,
                const MultiArrayDouble& p_out, double t) {

        LOG_F(INFO, "Recording velocity, pressure and force.");
        // 网格参数
        int3    dim_center = {_Nx + 2, _Ny + 2, _Nz + 2};
        double3 origin     = {-0.5 * _dx, -0.5 * _dy, -0.5 * _dz};
        double3 dh         = {_dx, _dy, _dz};

        auto velocity_center = algebra::create_multi_array<3, double3>({_Nx + 2, _Ny + 2, _Nz + 2});
        auto force_center    = algebra::create_multi_array<3, double3>({_Nx + 2, _Ny + 2, _Nz + 2});
        stagger_to_center<3, double, double3>(velocity_center, u_out, v_out, w_out, {_Nx, _Ny, _Nz});
        stagger_to_center<3, double, double3>(force_center, fu_out, fv_out, fw_out, {_Nx, _Ny, _Nz});

        // double3 velocity_sample = velocity_center[_Nx / 2][_Ny / 2][_Nz / 2];
        // double3 velocity_sample1 = velocity_center[_Nx / 2 + 1][_Ny / 2 + 1][_Nz / 2 + 1];
        // printf("Point = (%f, %f, %.16f), Velocity = (%f, %f, %.16f)\n",
        //         _Lx/2, _Ly/2, _Lz/2,
        //         (velocity_sample.x + velocity_sample1.x)*0.5,
        //         (velocity_sample.y + velocity_sample1.y)*0.5,
        //         (velocity_sample.z + velocity_sample1.z)*0.5);
        writer.write(dim_center, origin, dh, algebra::flatten(velocity_center), algebra::flatten(force_center),
                     algebra::flatten(p_out), t);
    }

    double kinematic_energy(const MultiArrayDouble& u_out, const MultiArrayDouble& v_out,
                            const MultiArrayDouble& w_out) {
        auto velocity_center = algebra::create_multi_array<3, double3>({_Nx + 2, _Ny + 2, _Nz + 2});
        stagger_to_center<3, double, double3>(velocity_center, u_out, v_out, w_out, {_Nx, _Ny, _Nz});
        double energy = 0.0;
        for (size_t i = 0; i < _Nx + 2; i++)
            for (size_t j = 0; j < _Ny + 2; j++)
                for (size_t k = 0; k < _Nz + 2; k++) {
                    energy += velocity_center[i][j][k].x * velocity_center[i][j][k].x
                              + velocity_center[i][j][k].y * velocity_center[i][j][k].y
                              + velocity_center[i][j][k].z * velocity_center[i][j][k].z;
                }
        energy = 0.5 * energy * _rho * _dx * _dy * _dz;
        return energy;
    }

    auto solve() {
        LOG_F(INFO, "Solving N-S quations at t = %f.", _t);
        // 获取边界条件和右端项, 包括:
        // f1,f2,f3, un, vn, wn, pn, uh, vh, wh, ph
        // NOTE: (一阶精度的投影方法不需要pn)
        set_boundary_conditions();
        set_source_terms();
        solve_one_step(uh, vh, wh, ph, u_, v_, w_, p_, f1, f2, f3, s, un, vn, wn, pn, ubv, vbv, wbv, pbv, ubt, vbt, wbt,
                       pbt, bu, bv, bw, bp, ru, rv, rw, rp, _Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz, _rho, _mu, max_iters,
                       tolerance);

        pn = ph;
        un = uh;
        vn = vh;
        wn = wh;
        post_process(0);
        return std::make_tuple(std::cref(uh), std::cref(vh), std::cref(wh), std::cref(ph));
    }
}; // class StokesFlow
} // namespace stokes_flow

template <typename DerivedStokesSolver>
class StokesFlowFactory {
  public:
    // 我们一般使用不带解析解的算例，如果使用 json 文件创建的 StokesFlow 对象会附
    // 带一个解析解，正确设置边界条件和源项，可以验证求解器的精度。
    static std::shared_ptr<stokes_flow::StokesFlow<3>> create(const std::array<int, DIM>& dim, int Nt,
                                                              const std::array<double, DIM>& L, double T, double rho,
                                                              double mu, std::string output_file = "fluid/data.pvd",
                                                              std::string input_demo_file = "") {
        std::shared_ptr<NavierStokesDemo> ns_demo = nullptr;

        if (input_demo_file.empty()) {
            ns_demo = std::make_shared<NavierStokesDemo>();
        } else {
            ns_demo = std::make_shared<NavierStokesDemo>(JsonFile(input_demo_file));
        }
        ns_demo->set(dim[0], dim[1], dim[2], Nt, L[0], L[1], L[2], T, rho, mu);
        return std::make_shared<DerivedStokesSolver>(ns_demo, output_file);
    }

}; // class StokesFlowFactory
#endif

// 设置边界条件类型和边界条件的值有两种方式：
// 1. 分别设置每条边界
//      update_boundary_values(ubt,ubv,vbt,vbv,pbt,pbv)
//      update_boundary_values(ubt,ubv,vbt,vbv,wbt,wbv,pbt,pbv)
// 2. 使用向量类型，只使用一个函数:
//      update_boundary_values(bt,bv),
//
// 求解函数需要有两个接口，一个是带有源项的求解器，一个是不带源项的求解器。
//
// solve_one_step(fn, un, u, p)
// solve_one_step(fn, un, u, p, s)
// 其中，fn, un, u 是向量变量，为(u,v)或者(u,v,w)。
//
// void solveOneStep(
//         const std::vector<double3> &vector_fn,
//         const std::vector<double3> &vector_un,
//         std::vector<double3> &vector_u,
//         std::vector<double> &vector_p)
//
// void solveOneStep(
//         const std::vector<double3> &vector_fn,
//         const std::vector<double3> &vector_un,
//         std::vector<double3> &vector_u,
//         std::vector<double> &vector_p,
//         std::vector<double> &vector_s)
//
//
