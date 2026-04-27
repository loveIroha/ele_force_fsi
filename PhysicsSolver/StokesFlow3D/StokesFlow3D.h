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
#include <PhysicsSolver/StokesFlow3D/NavierStokesDemo_deprecated.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>
#include <PhysicsSolver/StokesFlow3D/Projection3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityU3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityV3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityW3D.h>
#include <io/ScopeProfiler.h>
#include <io/loguru.hpp>
#include <io/vector_io.h>
#include <io/writeVTK.h>

// acceleration
#include <PhysicsSolver/StokesFlow3D/Pressure3D_kokkos.h>
#include <advection.h>

namespace stokes_flow {
const int INTERIOR  = 0;
const int DIRICHLET = 1;
const int NEUMANN   = 2;

template <int DIM, class enable = void>
class StokesFlow;

template <int DIM>
class StokesFlow<DIM, std::enable_if_t<DIM == 3>> {
  public:
    int                  _Nx{1}, _Ny{1}, _Nz{1}, _Nt{1};
    std::array<int, DIM> _dim;
    double               _dx, _dy, _dz, _dt;
    double               _Lx{0.0}, _Ly{0.0}, _Lz{0.0}, _T{0.0};
    double               _rho, _mu, _t;

    bool      semi_lagrange = true;
    VTIWriter writer;

    std::array<int, 6> all_boundary_type   = {1, 1, 1, 1, 1, 1};
    std::array<int, 6> all_boundary_type_p = {2, 2, 2, 2, 2, 2};

    int    max_iters = 1e5;
    double tolerance = 1e-7;
    // double tolerance = std::sqrt(std::numeric_limits<double>::epsilon());

    std::shared_ptr<NavierStokesDemo> ns_demo = nullptr;

    using MultiArrayInt    = algebra::MultiArrayType<DIM, int>;
    using MultiArrayDouble = algebra::MultiArrayType<DIM, double>;

    MultiArrayDouble f1, f2, f3, s;
    MultiArrayDouble un, vn, wn, pn;
    MultiArrayDouble u_, v_, w_, p_;
    MultiArrayDouble uh, vh, wh, ph;
    MultiArrayDouble ru, rv, rw, rp;
    MultiArrayDouble bu, bv, bw, bp;
    MultiArrayDouble ubv, vbv, wbv, pbv;
    MultiArrayInt    ubt, vbt, wbt, pbt;

    std::unique_ptr<LinearAdvection> advection = nullptr;

  public:
    std::array<int, DIM> get_dim_p() const {
        auto dim_p = _dim;
        for (int i = 0; i < DIM; ++i) {
            dim_p[i] = _dim[i] + 2;
        }
        return dim_p;
    }

    std::array<int, DIM> get_dim_u() const {
        CHECK_F(DIM >= 2, "DIM can be 1, 2, 3.");
        auto dim_u = get_dim_p();
        dim_u[0] -= 1;
        return dim_u;
    }

    std::array<int, DIM> get_dim_v() const {
        CHECK_F(DIM >= 2, "DIM can be 2,3.");
        auto dim_v = get_dim_p();
        dim_v[1] -= 1;
        return dim_v;
    }

    std::array<int, DIM> get_dim_w() const {
        CHECK_F(DIM == 3, "DIM must be 3.");
        auto dim_w = get_dim_p();
        dim_w[2] -= 1;
        return dim_w;
    }

    void set_source(const std::vector<double>& eulerian_force_u, const std::vector<double>& eulerian_force_v,
                    const std::vector<double>& eulerian_force_w) {
        f1 = algebra::ripple(eulerian_force_u, f1.size(),
                             f1[0].size()); // 获取上一时刻力场
        f2 = algebra::ripple(eulerian_force_v, f2.size(),
                             f2[0].size()); // 获取上一时刻力场
        f3 = algebra::ripple(eulerian_force_w, f3.size(),
                             f3[0].size()); // 获取上一时刻力场
    }

    auto get_source() const { return std::make_tuple(std::cref(f1), std::cref(f2), std::cref(f3)); }

    void set_velocity(const std::vector<double>& eulerian_velocity_u, const std::vector<double>& eulerian_velocity_v,
                      const std::vector<double>& eulerian_velocity_w) {
        un = algebra::ripple(eulerian_velocity_u, un.size(),
                             un[0].size()); // 获取上一时刻力场
        vn = algebra::ripple(eulerian_velocity_v, vn.size(),
                             vn[0].size()); // 获取上一时刻力场
        wn = algebra::ripple(eulerian_velocity_w, wn.size(),
                             wn[0].size()); // 获取上一时刻力场
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

    StokesFlow(int Nt, const std::array<int, DIM>& dim, const std::array<double, DIM>& L, double T, double rho,
               double mu, std::string output_file = "fluid/data.pvd")
        : _Nt(Nt), _dim(dim), _dt(T / Nt), _T{T}, _rho(rho), _mu(mu), writer{output_file} {
        ScopeProfiler _{__func__};

        _Nx = dim[0];
        _Ny = dim[1];
        _Lx = L[0];
        _Ly = L[1];
        _dx = _Lx / _Nx;
        _dy = _Ly / _Ny;

        if (DIM == 3) {
            _Nz = dim[2];
            _Lz = L[2];
            _dz = _Lz / _Nz;
        }

        CHECK_F(_Nx * _Ny * _Nz > 0, "Variable dim makes no sense.");
        advection = std::make_unique<LinearAdvection>(_Nx, _Ny, _Nz, _dt, _Lx, _Ly, _Lz);

        un = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        vn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        wn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        pn = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});

        u_ = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        v_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        w_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        p_ = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});

        f1 = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        f2 = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        f3 = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        s  = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});

        bu = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        bv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        bw = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        bp = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        ru = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        rv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        rw = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        rp = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        ubv = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        vbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        wbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        pbv = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        ubt = algebra::create_multi_array<3, int>({_Nx + 1, _Ny + 2, _Nz + 2});
        vbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 1, _Nz + 2});
        wbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 2, _Nz + 1});
        pbt = algebra::create_multi_array<3, int>({_Nx + 2, _Ny + 2, _Nz + 2});

        uh = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
        vh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
        wh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
        ph = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});

        JsonFile json_file("/home/kokkos/npuheart/features/finite_difference/3D/stokes_demo.json");
        ns_demo = std::make_shared<NavierStokesDemo>(json_file);
        ns_demo->set(_Nx, _Ny, _Nz, _Nt, _Lx, _Ly, _Lz, _T, _rho, _mu);

        LOG_F(INFO, "Lx : %.4f, Ly : %.4f, Lz : %.4f, T  : %.4f, rho : %.4f, mu : %.4f.", _Lx, _Ly, _Lz, _T, _rho, _mu);
        LOG_F(INFO, "_Nx : %04d, _Ny : %04d, _Nz : %04d, Nt : %08d, max_iters : %05d.", _Nx, _Ny, _Nz, _Nt, max_iters);
        LOG_F(INFO, "dx : %.6f, dy : %.6f, dz : %.6f, dt : %.8f.", _dx, _dy, _dz, _dt);
    }

    virtual void initial_values() {
        ns_demo->get_array_u(un, 0.0);
        ns_demo->get_array_v(vn, 0.0);
        ns_demo->get_array_w(wn, 0.0);
    }

    virtual void set_source_terms() {
        ns_demo->get_array_f1(f1, _t);
        ns_demo->get_array_f2(f2, _t);
        ns_demo->get_array_f3(f3, _t);
    }

    void initial_boundary_conditions() {
        ns_demo->get_boundary_type_u(ubt, all_boundary_type);
        ns_demo->get_boundary_type_v(vbt, all_boundary_type);
        ns_demo->get_boundary_type_w(wbt, all_boundary_type);
        ns_demo->get_boundary_type_p(pbt, all_boundary_type_p);
    }
    virtual void set_boundary_conditions() {
        ns_demo->get_boundary_values_u(ubv, ubt, _t);
        ns_demo->get_boundary_values_v(vbv, vbt, _t);
        ns_demo->get_boundary_values_w(wbv, wbt, _t);
        ns_demo->get_boundary_values_p(pbv, pbt, _t);
    }

    virtual void post_process(int i) {
        // 验证散度为0
        [[maybe_unused]] double div
            = projection_3D::calculate_velocity_divergence(un, vn, wn, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz);
        // printf("div = %.20e\n", div);

        // 计算误差
        if (i == _Nt) {
            {
                auto exact = ns_demo->get_array_p(_t);
                auto eh    = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 2});
                algebra::axpy(-1.0, exact, ph, eh);
                algebra::zero_boundary(eh);
                double sum_eh_squared = algebra::norm(eh) * std::sqrt(_dx * _dy * _dz);
                LOG_F(INFO, "p sum_eh_squared = %.20e", sum_eh_squared);
            }
            {
                auto exact = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
                ns_demo->get_array_u(exact, _t);
                auto eh = algebra::create_multi_array<3, double>({_Nx + 1, _Ny + 2, _Nz + 2});
                algebra::axpy(-1.0, exact, un, eh);
                algebra::zero_boundary(eh);
                double sum_eh_squared = algebra::norm(eh) * std::sqrt(_dx * _dy * _dz);
                LOG_F(INFO, "u sum_eh_squared = %.20e", sum_eh_squared);
            }
            {
                auto exact = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
                ns_demo->get_array_v(exact, _t);
                auto eh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 1, _Nz + 2});
                algebra::axpy(-1.0, exact, vn, eh);
                algebra::zero_boundary(eh);
                double sum_eh_squared = algebra::norm(eh) * std::sqrt(_dx * _dy * _dz);
                LOG_F(INFO, "v sum_eh_squared = %.20e", sum_eh_squared);
            }
            {
                auto exact = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
                ns_demo->get_array_w(exact, _t);
                auto eh = algebra::create_multi_array<3, double>({_Nx + 2, _Ny + 2, _Nz + 1});
                algebra::axpy(-1.0, exact, wn, eh);
                algebra::zero_boundary(eh);
                double sum_eh_squared = algebra::norm(eh) * std::sqrt(_dx * _dy * _dz);
                LOG_F(INFO, "w sum_eh_squared = %.20e", sum_eh_squared);
            }
        }
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
                        MultiArrayDouble& rp, int _Nx, int _Ny, int _Nz, int _Nt, double _Lx, double _Ly, double _Lz,
                        double _T, double _rho, double _mu, int max_iters, double tolerance) {
        ScopeProfiler _{__func__};
        double        _dt = _T / _Nt;

        // 对流项
        v_ = vn;
        u_ = un;
        w_ = wn;
        if (semi_lagrange) { advection->advection(u_, v_, w_); }

        // bu bv bw
        //  计算右端项
        ns_demo->make_tentitive_ub(bu, u_, f1);
        ns_demo->make_tentitive_vb(bv, v_, f2);
        ns_demo->make_tentitive_wb(bw, w_, f3);

        // u_ v_ w_
        // 求解中间速度
        velocity_u_3D::solve_u(ru, u_, ubt, ubv, bu, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz, _mu, _rho, _dt, max_iters,
                               tolerance);
        velocity_v_3D::solve_v(rv, v_, vbt, vbv, bv, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz, _mu, _rho, _dt, max_iters,
                               tolerance);
        velocity_w_3D::solve_w(rw, w_, wbt, wbv, bw, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz, _mu, _rho, _dt, max_iters,
                               tolerance);

        // uh vh wh
        vh = v_;
        uh = u_;
        wh = w_;

        // ph
        // 考虑不可压条件，计算
        // TODO: 考虑散度不为零的情况
        projection_3D::make_pb(bp, u_, v_, w_, _Nx, _Ny, _Nz, _Nt, _Lx, _Ly, _Lz, _T, _rho, _mu);
        pressure_3D::solve_p(ph, rp, pbt, pbv, bp, _Nx, _Ny, _Nz, _Lx, _Ly, _Lz, max_iters, tolerance);
        projection_3D::correct_velocity(uh, vh, wh, u_, v_, w_, ph, _Nx, _Ny, _Nz, _Nt, _Lx, _Ly, _Lz, _T, _rho);
        LOG_F(INFO, "Fluid solver at t = %f, uh_norm2 = %.16e", _t, algebra::norm(uh));
    }

    void record(const MultiArrayDouble& u_out, const MultiArrayDouble& v_out, const MultiArrayDouble& w_out,
                const MultiArrayDouble& fu_out, const MultiArrayDouble& fv_out, const MultiArrayDouble& fw_out,
                const MultiArrayDouble& p_out, double t) {
        LOG_F(INFO, "Recording velocity, pressure and force.");

        int3    dim_center = {_Nx + 2, _Ny + 2, _Nz + 2};
        double3 origin     = {-0.5 * _Lx / _Nx, -0.5 * _Ly / _Ny, -0.5 * _Ly / _Ny};
        double3 dh         = {_Lx / _Nx, _Ly / _Ny, _Ly / _Ny};

        auto velocity_center = algebra::create_multi_array<3, double3>({_Nx + 2, _Ny + 2, _Nz + 2});
        auto force_center    = algebra::create_multi_array<3, double3>({_Nx + 2, _Ny + 2, _Nz + 2});
        stagger_to_center<3, double, double3>(velocity_center, u_out, v_out, w_out, {_Nx, _Ny, _Nz});
        stagger_to_center<3, double, double3>(force_center, fu_out, fv_out, fw_out, {_Nx, _Ny, _Nz});

        writer.write(dim_center, origin, dh, algebra::flatten(velocity_center), algebra::flatten(force_center),
                     algebra::flatten(p_out), t);
    }

    auto solve() {
        LOG_F(INFO, "Solving N-S quations at t = %f.", _t);
        // 获取边界条件和右端项
        // f1,f2,f3, un, vn, wn, uh, vh, wh, ph
        set_boundary_conditions();
        set_source_terms();
        solve_one_step(uh, vh, wh, ph, u_, v_, w_, p_, f1, f2, f3, s, un, vn, wn, pn, ubv, vbv, wbv, pbv, ubt, vbt, wbt,
                       pbt, bu, bv, bw, bp, ru, rv, rw, rp, _Nx, _Ny, _Nz, _Nt, _Lx, _Ly, _Lz, _T, _rho, _mu, max_iters,
                       tolerance);

        pn = ph;
        un = uh;
        vn = vh;
        wn = wh;
        post_process(0);
        return std::make_tuple(std::cref(uh), std::cref(vh), std::cref(wh), std::cref(ph));
    }

    int main_helmholtz() {
        initial_values();

        for (int i = 1; i <= _Nt; i++) {
            _t = i * _dt;
            printf("t = %f\n", _t);
            // 获取边界条件和右端项
            // f1,f2,f3, un, vn, wn, uh, vh, wh, ph

            // 对流项
            // NOTE: 对流项放在设置边界条件之前面
            if (semi_lagrange) { advection->advection(un, vn, wn); }

            set_boundary_conditions();
            set_source_terms();

            solve_one_step(uh, vh, wh, ph, u_, v_, w_, p_, f1, f2, f3, s, un, vn, wn, pn, ubv, vbv, wbv, pbv, ubt, vbt,
                           wbt, pbt, bu, bv, bw, bp, ru, rv, rw, rp, _Nx, _Ny, _Nz, _Nt, _Lx, _Ly, _Lz, _T, _rho, _mu,
                           max_iters, tolerance);

            pn = ph;
            un = uh;
            vn = vh;
            wn = wh;

            post_process(i);
        }

        return 0;
    }
};
} // namespace stokes_flow
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
