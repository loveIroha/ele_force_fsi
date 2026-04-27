/// @date 2023-06-16
/// @file BackgroundMesh2D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __BACKGROUND_MESH_2D_H__
#define __BACKGROUND_MESH_2D_H__

#include <MeshTools/MeshFunctionManager.h>

template <size_t DIM = 2>
class BackgroundMesh2D : public MeshFunctionManager {
  public:
    size_t _Nx{}, _Ny{}, _Nz{}, _Nt{};
    double _dx, _dy, _dz, _dt;
    double _Lx = 0.0, _Ly = 0.0, _Lz = 0.0;
    double _T, _rho, _mu;

    // 残留代码
    double3 origin{0.0, 0.0, 0.0};

    virtual int num_dofs() const override final {
        LOG_F(INFO,
              "The number of dofs is %zu, number dofs at each dimension is %zu "
              "%zu %zu.",
              _Nx * _Ny * _Nz, _Nx, _Ny, _Nz);
        return _Nx * _Ny * _Nz;
    }

    double3 get_h3() const { return make_double3(_dx, _dy, _dz); }
    int2    get_dim() const { return make_int2(_Nx + 2, _Ny + 2); }
    int2    get_dim_u() const { return make_int2(_Nx + 1, _Ny + 2); }
    int2    get_dim_v() const { return make_int2(_Nx + 2, _Ny + 1); }
    int2    get_dim_p() const { return make_int2(_Nx + 2, _Ny + 2); }
    size_t  get_size_u() const { return (_Nx + 1) * (_Ny + 2); }
    size_t  get_size_v() const { return (_Nx + 2) * (_Ny + 1); }
    size_t  get_size_p() const { return (_Nx + 2) * (_Ny + 2); }

  public:
    template <typename T>
    auto make_vector_2D(int N, int M) {
        std::vector<std::vector<T>> result;
        result.resize(N, std::vector<T>(M));
        return result;
    }

    BackgroundMesh2D(double dt, size_t _Nx, size_t _Ny, double Lx, double Ly, double T, double rho, double mu)
        : BackgroundMesh2D((size_t)T / dt, {_Nx, _Ny}, {Lx, Ly}, T, rho, mu) {}

    BackgroundMesh2D(size_t _Nt, size_t _Nx, size_t _Ny, double Lx, double Ly, double T, double rho, double mu)
        : BackgroundMesh2D(_Nt, {_Nx, _Ny}, {Lx, Ly}, T, rho, mu) {}

    BackgroundMesh2D(size_t _Nt, const std::array<size_t, DIM>& dim, const std::array<double, DIM>& L, double T,
                     double rho, double mu)
        : _Nt(_Nt), _dt(T / _Nt), _T(T), _rho(rho), _mu(mu) {
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

        printf("Lx : %.2f, Ly : %.2f, Lz : %.2f, T  : %.2f, rho : %.2f, mu : "
               "%.2f.\n",
               _Lx, _Ly, _Lz, _T, _rho, _mu);
        printf("_Nx : %04zu, _Ny : %04zu, _Nz : %04zu, Nt : %04zu.\n", _Nx, _Ny, _Nz, _Nt);
        printf("dx : %.2f, dy : %.2f, dz : %.2f, dt : %.2f.\n", _dx, _dy, _dz, _dt);
    }

    virtual std::vector<double2>& set_function(std::string         name,
                                               Double2FunctionType function_expression) override final {
        // Check if data already exists
        auto it = functions_data.find(name);
        if (it == functions_data.end()) {
            LOG_F(INFO,
                  "The function data named \"%s\" did not exist, so we have "
                  "created it.",
                  name.c_str());
            create_function<double2>(name);
            it = functions_data.find(name);
        }

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double2>&>(it->second);
        for (size_t i = 0; i < _Nx + 2; i++) {
            for (size_t j = 0; j < _Ny + 2; j++) {
                for (size_t k = 0; k < _Nz + 2; k++) {
                    // HACK : This is a hack to make the function data
                    // compatible with the mesh data LOG_F(WARNING, "Calculating
                    // the %dth, %dth, %dth point.", i, j, k);
                    auto position = make_double3((i - 0.5) * _dx, (j - 0.5) * _dy, (k - 0.5) * _dz);
                    function_data[i + (_Nx + 2) * j + (_Ny + 2) * (_Nz + 2) * k] = function_expression(position);
                }
            }
        }

        return boost::any_cast<std::vector<double2>&>(it->second);
    }

    std::vector<std::vector<double2>> cell_center_velocity(const std::vector<std::vector<double>>& eulerian_u,
                                                           const std::vector<std::vector<double>>& eulerian_v) {
        auto result = make_vector_2D<double2>(_Nx + 2, _Ny + 2);
        for (size_t i = 0; i < _Nx; i++) {
            for (size_t j = 0; j < _Ny; j++) {
                result[i + 1][j + 1].x = 0.5 * (eulerian_u[i][j + 1] + eulerian_u[i + 1][j + 1]);
                result[i + 1][j + 1].y = 0.5 * (eulerian_v[i + 1][j] + eulerian_v[i + 1][j + 1]);
            }
        }
        return result;
    }

    template <typename FuncType>
    auto set_function_on_centered_grid(FuncType function_expression) const {
        auto                dim_p = get_dim_p();
        std::vector<double> eulerian_p(dim_p.x * dim_p.y);

        for (int i = 0; i < dim_p.x; i++) {
            for (int j = 0; j < dim_p.y; j++) {
                double3 point{(i - 0.5) * _dx, (j - 0.5) * _dy, 0};
                auto    value               = function_expression(point);
                eulerian_p[i + j * dim_p.x] = value;
            }
        }
        return eulerian_p;
    }

    auto set_function_on_staggered_grid(Double2FunctionType function_expression) {
        auto                dim_u = get_dim_u();
        auto                dim_v = get_dim_v();
        std::vector<double> eulerian_velocity_u(dim_u.x * dim_u.y);
        std::vector<double> eulerian_velocity_v(dim_v.x * dim_v.y);

        for (int i = 0; i < dim_u.x; i++) {
            for (int j = 0; j < dim_u.y; j++) {
                double3 point{i * _dx, (j - 0.5) * _dy, 0};
                auto    value                        = function_expression(point);
                eulerian_velocity_u[i + j * dim_u.x] = value.x;
            }
        }

        for (int i = 0; i < dim_v.x; i++) {
            for (int j = 0; j < dim_v.y; j++) {
                double3 point{(i - 0.5) * _dx, j * _dy, 0};
                auto    value                        = function_expression(point);
                eulerian_velocity_v[i + j * dim_v.x] = value.y;
            }
        }
        return std::make_tuple(eulerian_velocity_u, eulerian_velocity_v);
    }
};

#endif