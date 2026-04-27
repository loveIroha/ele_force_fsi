#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "Pressure.h"
#include "SemiLagrange.h"
#include "TentitiveVelocityU.h"
#include "TentitiveVelocityV.h"
#include "header.h"

template <size_t DIM = 2>
class StokesFlow {
  public:
    size_t _Nx{}, _Ny{}, _Nz{}, _Nt{};
    double _dx, _dy, _dz, _dt;
    double _Lx = 0.0, _Ly = 0.0, _Lz = 0.0;
    double _rho, _mu, _T;

    std::vector<std::vector<int>>    ubt;
    std::vector<std::vector<double>> ubv;
    std::vector<std::vector<int>>    vbt;
    std::vector<std::vector<double>> vbv;
    std::vector<std::vector<int>>    pbt;
    std::vector<std::vector<double>> pbv;

    double EPSILON   = std::sqrt(std::numeric_limits<double>::epsilon());
    size_t max_iters = 1e5;

  public:
    StokesFlow(double dt, size_t _Nx, size_t _Ny, double Lx, double Ly, double T, double rho, double mu)
        : StokesFlow((size_t)T / dt, {_Nx, _Ny}, {Lx, Ly}, T, rho, mu) {}

    StokesFlow(size_t _Nt, size_t _Nx, size_t _Ny, double Lx, double Ly, double T, double rho, double mu)
        : StokesFlow(_Nt, {_Nx, _Ny}, {Lx, Ly}, T, rho, mu) {}

    StokesFlow(size_t _Nt, const std::array<size_t, DIM>& dim, const std::array<double, DIM>& L, double T, double rho,
               double mu)
        : _Nt(_Nt), _dt(T / _Nt), _rho(rho), _mu(mu), _T(T) {
        _Nx = dim[0];
        _Ny = dim[1];
        _Nz = 1;
        _Lx = L[0];
        _Ly = L[1];
        _Lz = 0;
        _dx = _Lx / _Nx;
        _dy = _Ly / _Ny;
        _dz = _Lz / _Nx;

        if (DIM == 3) {
            _Nz = dim[2];
            _Lz = L[2];
            _dz = _Lz / _Nz;
        }

        ubt = make_vector_2D<int>(_Nx + 1, _Ny + 2);
        ubv = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        vbt = make_vector_2D<int>(_Nx + 2, _Ny + 1);
        vbv = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        pbt = make_vector_2D<int>(_Nx + 2, _Ny + 2);
        pbv = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        printf("Lx : %.2f, Ly : %.2f, Lz : %.2f, T  : %.2f, rho : %.2f, mu : %.2f.\n", _Lx, _Ly, _Lz, _T, _rho, _mu);
        printf("_Nx : %04zu, _Ny : %04zu, _Nz : %04zu, Nt : %04zu, max_iters : %05zu.\n", _Nx, _Ny, _Nz, _Nt,
               max_iters);
        printf("dx : %.2f, dy : %.2f, dz : %.2f, dt : %.2f.\n", _dx, _dy, _dz, _dt);
    }

    // u_exact, uf, ubv, ubt
    // v_exact, vf, vbv, vbt
    // p_exact, pbv, pbt
    // t
    int compute_demo(std::vector<std::vector<double>>& u_exact, std::vector<std::vector<double>>& uf,
                     std::vector<std::vector<double>>& ubv, const std::vector<std::vector<int>>& ubt,
                     std::vector<std::vector<double>>& v_exact, std::vector<std::vector<double>>& vf,
                     std::vector<std::vector<double>>& vbv, const std::vector<std::vector<int>>& vbt,
                     std::vector<std::vector<double>>& p_exact, std::vector<std::vector<double>>& pbv,
                     const std::vector<std::vector<int>>& pbt, double t, double width, double height, double T,
                     double rho, double mu) {
        double dx = width / _Nx;
        double dy = height / _Ny;
        double dt = T / _Nt;

        // sympy
        auto fun_u = [mu, rho](double x, double y, double t) -> double {
            return -1.0L / 256.0L * pow(x, 2) * y * pow(x - 1, 2) * (y - 1) * (2 * y - 1) * exp(t);
        };
        auto fun_uf = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 256.0L)
                   * (2 * mu * (2 * y - 1)
                          * (3 * pow(x, 2) * pow(x - 1, 2)
                             + y * (y - 1) * (pow(x, 2) + 4 * x * (x - 1) + pow(x - 1, 2)))
                      - rho * pow(x, 2) * y * pow(x - 1, 2) * (y - 1) * (2 * y - 1) + 768 * pow(x, 2))
                   * exp(t);
        };
        auto fun_dudx = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 128.0L) * x * y * (-2 * x + 1) * (x - 1) * (y - 1) * (2 * y - 1) * exp(t);
        };
        auto fun_dudy = [mu, rho](double x, double y, double t) -> double {
            return -1.0L / 256.0L * pow(x, 2) * pow(x - 1, 2)
                   * (2 * y * (y - 1) + y * (2 * y - 1) + (y - 1) * (2 * y - 1)) * exp(t);
        };
        auto fun_v = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 256.0L) * x * pow(y, 2) * (x - 1) * (2 * x - 1) * pow(y - 1, 2) * exp(t);
        };
        auto fun_vf = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 256.0L) * (2 * x - 1)
                   * (-2 * mu
                          * (x * (x - 1) * (pow(y, 2) + 4 * y * (y - 1) + pow(y - 1, 2))
                             + 3 * pow(y, 2) * pow(y - 1, 2))
                      + rho * x * pow(y, 2) * (x - 1) * pow(y - 1, 2))
                   * exp(t);
        };
        auto fun_dvdx = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 256.0L) * pow(y, 2) * pow(y - 1, 2)
                   * (2 * x * (x - 1) + x * (2 * x - 1) + (x - 1) * (2 * x - 1)) * exp(t);
        };
        auto fun_dvdy = [mu, rho](double x, double y, double t) -> double {
            return (1.0L / 128.0L) * x * y * (x - 1) * (2 * x - 1) * (y - 1) * (2 * y - 1) * exp(t);
        };
        auto fun_p    = [mu, rho](double x, double y, double t) -> double { return (pow(x, 3) - 0.25) * exp(t); };
        auto fun_dpdx = [mu, rho](double x, double y, double t) -> double { return 3 * pow(x, 2) * exp(t); };
        auto fun_dpdy = [mu, rho](double x, double y, double t) -> double { return 0; };

        // compute u_exact, uf, ubv
        for (size_t i = 0; i < _Nx + 1; i++) {
            for (size_t j = 0; j < _Ny + 2; j++) {
                u_exact[i][j] = fun_u(dx * i, dy * j - 0.5 * dy, t);
                uf[i][j]      = fun_uf(dx * i, dy * j - 0.5 * dy, t);
            }
        }
        for (size_t i = 0; i < _Nx + 1; i++) {
            if (ubt[i][0] == DIRICHLET) ubv[i][0] = fun_u(dx * i, 0, t);
            if (ubt[i][0] == NEUMANN) ubv[i][0] = fun_dudy(dx * i, 0, t);
            if (ubt[i][_Ny + 1] == DIRICHLET) ubv[i][_Ny + 1] = fun_u(dx * i, 1, t);
            if (ubt[i][_Ny + 1] == NEUMANN) ubv[i][_Ny + 1] = fun_dudy(dx * i, 1, t);
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            if (ubt[0][j] == DIRICHLET) ubv[0][j] = fun_u(0, dy * j - 0.5 * dy, t);
            if (ubt[0][j] == NEUMANN) ubv[0][j] = fun_dudx(0, dy * j - 0.5 * dy, t);
            if (ubt[_Nx][j] == DIRICHLET) ubv[_Nx][j] = fun_u(1, dy * j - 0.5 * dy, t);
            if (ubt[_Nx][j] == NEUMANN) ubv[_Nx][j] = fun_dudx(1, dy * j - 0.5 * dy, t);
        }

        // compute v_exact, vf, vbv
        for (size_t i = 0; i < _Nx + 2; i++) {
            for (size_t j = 0; j < _Ny + 1; j++) {
                v_exact[i][j] = fun_v(dx * (i - 0.5), dy * j, t);
                vf[i][j]      = fun_vf(dx * (i - 0.5), dy * j, t);
            }
        }
        for (size_t i = 1; i < _Nx + 1; i++) {
            if (vbt[i][0] == DIRICHLET) vbv[i][0] = fun_v(dx * i - 0.5 * dx, 0, t);
            if (vbt[i][0] == NEUMANN) vbv[i][0] = fun_dvdy(dx * i - 0.5 * dx, 0, t);
            if (vbt[i][_Ny] == DIRICHLET) vbv[i][_Ny] = fun_v(dx * i - 0.5 * dx, 1, t);
            if (vbt[i][_Ny] == NEUMANN) vbv[i][_Ny] = fun_dvdy(dx * i - 0.5 * dx, 1, t);
        }
        for (size_t j = 0; j < _Ny + 1; j++) {
            if (vbt[0][j] == DIRICHLET) vbv[0][j] = fun_v(0, dy * j, t);
            if (vbt[0][j] == NEUMANN) vbv[0][j] = fun_dvdx(0, dy * j, t);
            if (vbt[_Nx + 1][j] == DIRICHLET) vbv[_Nx + 1][j] = fun_v(1, dy * j, t);
            if (vbt[_Nx + 1][j] == NEUMANN) vbv[_Nx + 1][j] = fun_dvdx(1, dy * j, t);
        }
        // compute p_exact, pbv
        for (size_t i = 0; i < _Nx + 2; i++) {
            for (size_t j = 0; j < _Ny + 2; j++) {
                p_exact[i][j] = fun_p(dx * i - 0.5 * dx, dy * j - 0.5 * dy, t);
            }
        }

        for (size_t i = 1; i < _Nx + 1; i++) {
            if (pbt[i][0] == DIRICHLET) pbv[i][0] = fun_p(dx * i - 0.5 * dx, 0, t);
            if (pbt[i][0] == NEUMANN) pbv[i][0] = fun_dpdy(dx * i - 0.5 * dx, 0, t);
            if (pbt[i][_Ny + 1] == DIRICHLET) pbv[i][_Ny + 1] = fun_p(dx * i - 0.5 * dx, 1, t);
            if (pbt[i][_Ny + 1] == NEUMANN) pbv[i][_Ny + 1] = fun_dpdy(dx * i - 0.5 * dx, 1, t);
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            if (pbt[0][j] == DIRICHLET) pbv[0][j] = fun_p(0, dy * j - 0.5 * dy, t);
            if (pbt[0][j] == NEUMANN) pbv[0][j] = fun_dpdx(0, dy * j - 0.5 * dy, t);
            if (pbt[_Nx + 1][j] == DIRICHLET) pbv[_Nx + 1][j] = fun_p(0, dy * j - 0.5 * dy, t);
            if (pbt[_Nx + 1][j] == NEUMANN) pbv[_Nx + 1][j] = fun_dpdx(0, dy * j - 0.5 * dy, t);
        }
        return 0;
    }

    int compute_boundary_conditions(int ubt_edge[], int vbt_edge[], int pbt_edge[], double velocity_u[],
                                    double velocity_v[], double pressure[]) {
        compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, ubt, vbt, pbt);
        compute_boundary_conditions(velocity_u, velocity_v, pressure, ubv, vbv, pbv);
        return 0;
    }

    int compute_boundary_conditions(double velocity_u[], double velocity_v[], double pressure[],
                                    std::vector<std::vector<double>>& ubv, std::vector<std::vector<double>>& vbv,
                                    std::vector<std::vector<double>>& pbv) {
        for (size_t i = 1; i < _Nx + 1; i++) {
            vbv[i][0]   = velocity_v[0];
            vbv[i][_Ny] = velocity_v[1];
        }
        for (size_t j = 0; j < _Ny + 1; j++) {
            vbv[0][j]       = velocity_v[2];
            vbv[_Nx + 1][j] = velocity_v[3];
        }
        for (size_t i = 0; i < _Nx + 1; i++) {
            ubv[i][0]       = velocity_u[0];
            ubv[i][_Ny + 1] = velocity_u[1];
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            ubv[0][j]   = velocity_u[2];
            ubv[_Nx][j] = velocity_u[3];
        }
        for (size_t i = 1; i < _Nx + 1; i++) {
            pbv[i][0]       = pressure[0];
            pbv[i][_Ny + 1] = pressure[1];
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            pbv[0][j]       = pressure[2];
            pbv[_Nx + 1][j] = pressure[3];
        }
        return 0;
    }

    int compute_boundary_conditions(int ubt_edge[], int vbt_edge[], int pbt_edge[], std::vector<std::vector<int>>& ubt,
                                    std::vector<std::vector<int>>& vbt, std::vector<std::vector<int>>& pbt) {
        for (size_t i = 1; i < _Nx + 1; i++) {
            vbt[i][0]   = vbt_edge[0];
            vbt[i][_Ny] = vbt_edge[1];
        }
        for (size_t j = 0; j < _Ny + 1; j++) {
            vbt[0][j]       = vbt_edge[2];
            vbt[_Nx + 1][j] = vbt_edge[3];
        }
        for (size_t i = 0; i < _Nx + 1; i++) {
            ubt[i][0]       = ubt_edge[0];
            ubt[i][_Ny + 1] = ubt_edge[1];
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            ubt[0][j]   = ubt_edge[2];
            ubt[_Nx][j] = ubt_edge[3];
        }
        for (size_t i = 1; i < _Nx + 1; i++) {
            pbt[i][0]       = pbt_edge[0];
            pbt[i][_Ny + 1] = pbt_edge[1];
        }
        for (size_t j = 1; j < _Ny + 1; j++) {
            pbt[0][j]       = pbt_edge[2];
            pbt[_Nx + 1][j] = pbt_edge[3];
        }
        return 0;
    }

    int test_main() {
        double t = 0;

        double T   = _T;
        double rho = _rho;
        double mu  = _mu;

        double width  = _Lx;
        double height = _Ly;

        double dx = _dx;
        double dy = _dy;
        double dt = _dt;

        auto u_exact = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto u       = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto u_      = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto un      = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto uf      = make_vector_2D<double>(_Nx + 1, _Ny + 2);

        auto v_exact = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto v       = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto v_      = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto vn      = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto vf      = make_vector_2D<double>(_Nx + 2, _Ny + 1);

        auto p_exact = make_vector_2D<double>(_Nx + 2, _Ny + 2);
        auto p       = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        // 设置边界类型
        int ubt_edge[4] = {1, 1, 1, 1};
        int vbt_edge[4] = {1, 1, 1, 1};
        int pbt_edge[4] = {2, 2, 2, 2};
        compute_boundary_conditions(ubt_edge, vbt_edge, pbt_edge, ubt, vbt, pbt);

        // 计算un 和 vn
        compute_demo(un, uf, ubv, ubt, vn, vf, vbv, vbt, p_exact, pbv, pbt, t, width, height, T, rho, mu);

        test_stokes(u_exact, u, u_, un, ubt, ubv, uf, v_exact, v, v_, vn, vbt, vbv, vf, p_exact, p, pbt, pbv);
        return 0;
    }

    void test_stokes(std::vector<std::vector<double>>& u_exact, std::vector<std::vector<double>>& u,
                     std::vector<std::vector<double>>& u_, std::vector<std::vector<double>>& un,
                     std::vector<std::vector<int>>& ubt, std::vector<std::vector<double>>& ubv,
                     std::vector<std::vector<double>>& uf, std::vector<std::vector<double>>& v_exact,
                     std::vector<std::vector<double>>& v, std::vector<std::vector<double>>& v_,
                     std::vector<std::vector<double>>& vn, std::vector<std::vector<int>>& vbt,
                     std::vector<std::vector<double>>& vbv, std::vector<std::vector<double>>& vf,
                     std::vector<std::vector<double>>& p_exact, std::vector<std::vector<double>>& p,
                     std::vector<std::vector<int>>& pbt, std::vector<std::vector<double>>& pbv) {
        double t = 0;

        double width  = _Lx;
        double height = _Ly;
        double T      = _T;
        double rho    = _rho;
        double mu     = _mu;

        double dx = width / _Nx;
        double dy = height / _Ny;
        double dt = T / _Nt;

        for (size_t step = 1; step < 1000000000; step++) {
            t = step * dt;

            printf("step: %05zu, time: %.5f\n\n", step, t);

            // 计算 uf, ubt, ubv,
            // 计算 vf, vbt, vbv,
            // 计算 pbt, pbv
            compute_demo(u_exact, uf, ubv, ubt, v_exact, vf, vbv, vbt, p_exact, pbv, pbt, t, width, height, T, rho, mu);

            // 计算 u_, v_
            u_ = un;
            v_ = vn;

            // 计算出 u 和 v
            solve_one_step(u, v, p, uf, vf, un, vn);

            // 将 u, v 赋值给 un, vn
            un                = u;
            vn                = v;
            double l2_error_u = squared_distance(un, u_exact);
            double l2_error_v = squared_distance(vn, v_exact);

            printf("L2 error of u: %f\n", std::sqrt(l2_error_u * dx * dy));
            printf("L2 error of v: %f\n", std::sqrt(l2_error_v * dx * dy));

            if (t > T - EPSILON) break;
        }
    }

    int solve_one_step(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v,
                       std::vector<std::vector<double>>& p, const std::vector<std::vector<double>>& uf,
                       const std::vector<std::vector<double>>& vf, const std::vector<std::vector<double>>& un,
                       const std::vector<std::vector<double>>& vn) {
        auto u_ = un;
        auto v_ = vn;

        solve_one_step(u, u_, un, ubt, ubv, uf, v, v_, vn, vbt, vbv, vf, p, pbt, pbv, _Lx, _Ly, _T, _rho, _mu,
                       max_iters);
        return 0;
    }

    int solve_one_step(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v,
                       std::vector<std::vector<double>>& p, const std::vector<std::vector<double>>& uf,
                       const std::vector<std::vector<double>>& vf, const std::vector<std::vector<double>>& un,
                       const std::vector<std::vector<double>>& vn, const std::vector<std::vector<double>>& s) {
        auto u_ = un;
        auto v_ = vn;

        solve_one_step(u, u_, un, ubt, ubv, uf, v, v_, vn, vbt, vbv, vf, p, pbt, pbv, s, _Lx, _Ly, _T, _rho, _mu,
                       max_iters);
        return 0;
    }

    int solve_one_step(std::vector<double>& U, std::vector<double>& P, const std::vector<double>& F,
                       const std::vector<double>& UN) {
        // 将U拆分成 u, v, w
        // 将F拆分成 uf, vf, wf
        // 将UN拆分成 un, vn, wn
        auto u  = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto uf = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto un = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto v  = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto vf = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto vn = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        auto p  = make_vector_2D<double>(_Nx + 2, _Ny + 2);

        auto data_u  = &(U.data()[0]);
        auto data_un = &(UN.data()[0]);
        auto data_uf = &(F.data()[0]);
        for (size_t i = 0; i < _Nx + 1; i++) {
            for (size_t j = 0; j < _Ny + 2; j++) {
                u[i][j]  = data_u[i + (_Nx + 1) * j];
                un[i][j] = data_un[i + (_Nx + 1) * j];
                uf[i][j] = data_uf[i + (_Nx + 1) * j];
            }
        }
        auto data_v  = &(U.data()[(_Nx + 1) * (_Ny + 2)]);
        auto data_vn = &(UN.data()[(_Nx + 1) * (_Ny + 2)]);
        auto data_vf = &(F.data()[(_Nx + 1) * (_Ny + 2)]);
        for (size_t i = 0; i < _Nx + 2; i++) {
            for (size_t j = 0; j < _Ny + 1; j++) {
                v[i][j]  = data_v[i + (_Nx + 2) * j];
                vn[i][j] = data_vn[i + (_Nx + 2) * j];
                vf[i][j] = data_vf[i + (_Nx + 2) * j];
            }
        }
        for (size_t i = 0; i < _Nx + 2; i++) {
            for (size_t j = 0; j < _Ny + 1; j++) {
                p[i][j] = P[i * (_Nx + 2) + j];
            }
        }
        return solve_one_step(u, v, p, uf, vf, un, vn);
    }

    int correct_u(std::vector<std::vector<double>>& u, const std::vector<std::vector<double>>& u_,
                  const std::vector<std::vector<double>>& p, double width, double height, double T, double rho,
                  double mu) {
        double dx = width / _Nx;
        // double dy = height / _Ny;
        double dt = T / _Nt;
        for (size_t i = 0; i < _Nx + 1; i++) {
            for (size_t j = 1; j < _Ny + 1; j++) {
                u[i][j] = u_[i][j] - dt / rho / dx * (p[i + 1][j] - p[i][j]);
            }
            u[i][0]       = u_[i][0];
            u[i][_Ny + 1] = u_[i][_Ny + 1];
        }
        return 0;
    }

    int correct_v(std::vector<std::vector<double>>& v, const std::vector<std::vector<double>>& v_,
                  const std::vector<std::vector<double>>& p, double width, double height, double T, double rho,
                  double mu) {
        // double dx = width / _Nx;
        double dy = height / _Ny;
        double dt = T / _Nt;
        for (size_t i = 1; i < _Nx + 1; i++) {
            for (size_t j = 0; j < _Ny + 1; j++) {
                v[i][j] = v_[i][j] - dt / rho / dy * (p[i][j + 1] - p[i][j]);
            }
        }
        for (size_t j = 0; j < _Ny + 1; j++) {
            v[0][j]       = v_[0][j];
            v[_Nx + 1][j] = v_[_Nx + 1][j];
        }
        return 0;
    }

    void solve_one_step(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& u_,
                        const std::vector<std::vector<double>>& un, const std::vector<std::vector<int>>& ubt,
                        const std::vector<std::vector<double>>& ubv, const std::vector<std::vector<double>>& uf,
                        std::vector<std::vector<double>>& v, std::vector<std::vector<double>>& v_,
                        const std::vector<std::vector<double>>& vn, const std::vector<std::vector<int>>& vbt,
                        const std::vector<std::vector<double>>& vbv, const std::vector<std::vector<double>>& vf,
                        std::vector<std::vector<double>>& p, const std::vector<std::vector<int>>& pbt,
                        const std::vector<std::vector<double>>& pbv, double width, double height, double T, double rho,
                        double mu, int max_iters) {
        // 反向追踪出发点的速度
        auto u_d = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto v_d = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        semi_lagrange::trace(un, vn, u_d, v_d, _dt, _dx, _dy, _Nx, _Ny);

        tentitive_velocity_u(uf, u_d, u_, ubt, ubv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);
        tentitive_velocity_v(vf, v_d, v_, vbt, vbv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // // 下面是没有对流项的方程
        // tentitive_velocity_u(uf, un, u_, ubt, ubv, width, height, T, rho, mu,
        // max_iters, _Nx, _Ny, _Nt); tentitive_velocity_v(vf, vn, v_, vbt, vbv,
        // width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // 计算 p
        poisson(u_, v_, p, pbt, pbv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // 计算 u, v
        correct_u(u, u_, p, width, height, T, rho, mu);
        correct_v(v, v_, p, width, height, T, rho, mu);
    }

    void solve_one_step(std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& u_,
                        const std::vector<std::vector<double>>& un, const std::vector<std::vector<int>>& ubt,
                        const std::vector<std::vector<double>>& ubv, const std::vector<std::vector<double>>& uf,
                        std::vector<std::vector<double>>& v, std::vector<std::vector<double>>& v_,
                        const std::vector<std::vector<double>>& vn, const std::vector<std::vector<int>>& vbt,
                        const std::vector<std::vector<double>>& vbv, const std::vector<std::vector<double>>& vf,
                        std::vector<std::vector<double>>& p, const std::vector<std::vector<int>>& pbt,
                        const std::vector<std::vector<double>>& pbv, const std::vector<std::vector<double>>& s,
                        double width, double height, double T, double rho, double mu, int max_iters) {
        // 反向追踪出发点的速度
        auto u_d = make_vector_2D<double>(_Nx + 1, _Ny + 2);
        auto v_d = make_vector_2D<double>(_Nx + 2, _Ny + 1);
        semi_lagrange::trace(un, vn, u_d, v_d, _dt, _dx, _dy, _Nx, _Ny);

        tentitive_velocity_u(uf, u_d, u_, ubt, ubv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);
        tentitive_velocity_v(vf, v_d, v_, vbt, vbv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // // 下面是没有对流项的方程
        // tentitive_velocity_u(uf, un, u_, ubt, ubv, width, height, T, rho, mu,
        // max_iters, _Nx, _Ny, _Nt); tentitive_velocity_v(vf, vn, v_, vbt, vbv,
        // width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // 计算 p
        poisson(u_, v_, p, s, pbt, pbv, width, height, T, rho, mu, max_iters, _Nx, _Ny, _Nt);

        // 计算 u, v
        correct_u(u, u_, p, width, height, T, rho, mu);
        correct_v(v, v_, p, width, height, T, rho, mu);
    }

    ~StokesFlow() {}
};

// 设置边界条件类型和边界条件的值
// update_boundary_values(ubt,ubv,vbt,vbv,pbt,pbv)
// update_boundary_values(ubt,ubv,vbt,vbv,wbt,wbv,pbt,pbv)
// 或者使用向量类型，只使用一个函数:
// update_boundary_values(bt,bv),
// 这个函数调用前面三个类型的变量。

// 需要两个函数，一个是带有源项的求解器，一个是不带源项的求解器。
//
// solve_one_step(fn, un, u, p, s)
// 其中，fn, un, u 是向量变量，为(u,v)或者(u,v,w)。

// void solveOneStep(
//         const std::vector<double3> &vector_fn,
//         const std::vector<double3> &vector_un,
//         std::vector<double3> &vector_u,
//         std::vector<double> &vector_p)
