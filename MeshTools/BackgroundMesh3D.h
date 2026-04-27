/// @date 2023-09-28
/// @file BackgroundMesh3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///       2023-10-05 处理针对三维问题的代码
///       2024-05-03 去除Nt变量，时间相关的变量只保留dt和Lt
///

#pragma once

#include <MeshTools/MeshFunctionManager.h>

// 定义一个概念，要求可调用对象的参数类型为 double3 并返回 double3
template <typename Callable>
concept FuncDouble3ToDouble3 = requires(Callable callable, const double3& arg) {
    { callable(arg) } -> std::same_as<double3>;
};

template <int DIM = 3>
class BackgroundMesh3D : public MeshFunctionManager {
  public:
    int                  _Nx{1}, _Ny{1}, _Nz{1};
    std::array<int, DIM> _dim{};
    double               _dx, _dy, _dz, _dt;
    double               _Lx{0.0}, _Ly{0.0}, _Lz{0.0}, _Lt{0.0};
    double               _rho, _mu, _t = 0.0;

    // 残留代码
    double3 origin{0.0, 0.0, 0.0};

    virtual int num_dofs() const override final {
        CHECK_F(false, "Not implemented.");
        return 0;
    }

    double3 get_dh() const { return make_double3(_dx, _dy, _dz); }
    int3    get_dim() const { return make_int3(_Nx + 2, _Ny + 2, _Nz + 2); }

    int3 get_dim_u() const { return make_int3(_Nx + 3, _Ny + 2, _Nz + 2); }
    int3 get_dim_v() const { return make_int3(_Nx + 2, _Ny + 3, _Nz + 2); }
    int3 get_dim_w() const { return make_int3(_Nx + 2, _Ny + 2, _Nz + 3); }
    int3 get_dim_p() const { return make_int3(_Nx + 2, _Ny + 2, _Nz + 2); }

    double3 get_origin_u() const { return make_double3(-_dx, -0.5 * _dy, -0.5 * _dz); }
    double3 get_origin_v() const { return make_double3(-0.5 * _dx, -_dy, -0.5 * _dz); }
    double3 get_origin_w() const { return make_double3(-0.5 * _dx, -0.5 * _dy, -_dz); }
    double3 get_origin_p() const { return make_double3(-0.5 * _dx, -0.5 * _dy, -0.5 * -_dz); }

    int get_size_u() const { return (_Nx + 3) * (_Ny + 2) * (_Nz + 2); }
    int get_size_v() const { return (_Nx + 2) * (_Ny + 3) * (_Nz + 2); }
    int get_size_w() const { return (_Nx + 2) * (_Ny + 2) * (_Nz + 3); }
    int get_size_p() const { return (_Nx + 2) * (_Ny + 2) * (_Nz + 2); }

  public:
    BackgroundMesh3D(int _Nt, const std::array<int, DIM>& dim, const std::array<double, DIM>& L, double T, double rho,
                     double mu)
        : _dt(T / _Nt), _Lt(T), _rho(rho), _mu(mu) {
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

        printf("Lx : %.2f, Ly : %.2f, Lz : %.2f, T  : %.2f, rho : %.2f, mu : %.2f.\n", _Lx, _Ly, _Lz, _Lt, _rho, _mu);
        printf("_Nx : %04d, _Ny : %04d, _Nz : %04d, Nt : %04d.\n", _Nx, _Ny, _Nz, _Nt);
        printf("dx : %.2f, dy : %.2f, dz : %.2f, dt : %.2f.\n", _dx, _dy, _dz, _dt);
    }

    template <FuncDouble3ToDouble3 FuncType>
    auto set_function_on_staggered_grid(FuncType function_expression) {
        auto dim_u = get_dim_u();
        auto dim_v = get_dim_v();
        auto dim_w = get_dim_w();

        std::vector<double> eulerian_velocity_u(get_size_u());
        std::vector<double> eulerian_velocity_v(get_size_v());
        std::vector<double> eulerian_velocity_w(get_size_w());

        for (int i = 0; i < dim_u.x; i++) {
            for (int j = 0; j < dim_u.y; j++) {
                for (int k = 0; k < dim_u.z; k++) {
                    double3 point{(i - 1.0) * _dx, (j - 0.5) * _dy, (k - 0.5) * _dz};
                    auto    value                                                = function_expression(point);
                    eulerian_velocity_u[i + j * dim_u.x + k * dim_u.x * dim_u.y] = value.x;
                }
            }
        }

        for (int i = 0; i < dim_v.x; i++) {
            for (int j = 0; j < dim_v.y; j++) {
                for (int k = 0; k < dim_v.z; k++) {
                    double3 point{(i - 0.5) * _dx, (j - 1.0) * _dy, (k - 0.5) * _dz};
                    auto    value                                                = function_expression(point);
                    eulerian_velocity_v[i + j * dim_v.x + k * dim_v.x * dim_v.y] = value.y;
                }
            }
        }

        for (int i = 0; i < dim_w.x; i++) {
            for (int j = 0; j < dim_w.y; j++) {
                for (int k = 0; k < dim_w.z; k++) {
                    double3 point{(i - 0.5) * _dx, (j - 0.5) * _dy, (k - 1.0) * _dz};
                    auto    value                                                = function_expression(point);
                    eulerian_velocity_w[i + j * dim_w.x + k * dim_w.x * dim_w.y] = value.z;
                }
            }
        }
        return std::make_tuple(eulerian_velocity_u, eulerian_velocity_v, eulerian_velocity_w);
    }
};
