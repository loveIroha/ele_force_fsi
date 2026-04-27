/// @date 2023-12-13
/// @file Multigrid.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#pragma once
#include <PhysicsSolver/StokesFlow3D/some_headers.h>

#include <concepts>
#include <stdexcept>

namespace mg {

template <int DIM, class enable = void>
struct LevelInfo;
template <int DIM>
struct LevelInfo<DIM, std::enable_if_t<DIM == 2>> {
    double2 dh;
    int2    N;
};
template <int DIM>
struct LevelInfo<DIM, std::enable_if_t<DIM == 3>> {
    double3 dh;
    int3    N;
};

std::tuple<int, std::vector<LevelInfo<3>>> compute_multigrid_levels_3D(int Nx, int Ny, int Nz, double Lx, double Ly,
                                                                       double Lz) {
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    double dz = Lz / Nz;

    int num_levels = 0;

    std::vector<LevelInfo<3>> level_infos;

    while (true) {
        num_levels++;
        int3         N          = {Nx, Ny, Nz};
        double3      dh         = {dx, dy, dz};
        LevelInfo<3> level_info = {dh, N};
        level_infos.push_back(level_info);

        LOG_F(WATCH, "num_levels : %d, level_dh  : %2.4e, %2.4e, %2.4e", num_levels, dx, dy, dz);
        LOG_F(WATCH, "num_levels : %d, level_dim : %d, %d, %d", num_levels, Nx, Ny, Nz);

        if (Nx % 2 == 1 || Ny % 2 == 1 || Nz % 2 == 1) { break; }

        if (Nx * Ny * Nz < 65) { break; }

        Nx /= 2;
        Ny /= 2;
        Nz /= 2;
        dx *= 2;
        dy *= 2;
        dz *= 2;
    }

    if (num_levels <= 1) { throw std::runtime_error("the problem is too small for the multigrid solver."); }

    return {num_levels, level_infos};
}
template <typename T>
void copy(T& dst, const T& src) {
    dst = src;
}

template <typename T>
void zero(T& dst) {}

template <>
inline void zero<View3D<double>>(View3D<double>& a) {
    auto Nx = a.extent(0);
    auto Ny = a.extent(1);
    auto Nz = a.extent(2);

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
        "restrict", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { a(i, j, k) = 0.0; });
}

template <typename T>
void add(T& a, T& b) {}

template <>
inline void add<View3D<double>>(View3D<double>& a, View3D<double>& b) {
    auto Nx = a.extent(0);
    auto Ny = a.extent(1);
    auto Nz = a.extent(2);

    assert(a.extent(0) == b.extent(0) && a.extent(1) == b.extent(1) && a.extent(2) == b.extent(2));

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
    Kokkos::parallel_for(
        "restrict", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) { a(i, j, k) += b(i, j, k); });
}

template <typename T>
T create_array(int3 N) {
    CHECK_F(false, "Type %s not implemented!", typeid(T).name());
    return T{};
}
template <>
inline std::vector<double> create_array<std::vector<double>>(int3 N) {
    std::vector<double> a(0);
    return a;
}
template <>
inline std::vector<int> create_array<std::vector<int>>(int3 N) {
    std::vector<int> a(0);
    return a;
}
template <>
inline View3D<double> create_array<View3D<double>>(int3 N) {
    return View3D<double>("View3D<double>", N.x, N.y, N.z);
}
template <>
inline View3D<int> create_array<View3D<int>>(int3 N) {
    return View3D<int>("View3D<int>", N.x, N.y, N.z);
}
// 定义拉格朗日插值函数
template <typename T>
KOKKOS_INLINE_FUNCTION T L(int i, T x) {
    return (i == 0) * (1.0 - x) + (i == 1) * x;
}

// 修改 trilinear 函数以接受指针
template <typename T>
KOKKOS_INLINE_FUNCTION T trilinear(T x, T y, T z, const T* f) {
    T sum = 0.0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                sum += f[k * 4 + j * 2 + i] * L(i, x) * L(j, y) * L(k, z);
            }
        }
    }
    return sum;
}

template <typename T, template <typename> class GridArray>
class Multigrid {
  public:
    int3        _N;
    double3     _L;
    double3     _h;
    std::string _grid_type;
    double      _tol_relative;
    double      _tol_descending;
    int         _num_presmooth;
    int         _num_aftersmooth;
    int         _num_exactsmooth;
    int         _max_iteration;
    int         _N_coarsest;
    bool        PureNeumann = false;
    int3        pad         = {2, 2, 2};

    std::vector<LevelInfo<3>> _level_info;
    int                       _num_levels;

    GridArray<int>              _bct;
    std::vector<GridArray<T>>   multi_x;
    std::vector<GridArray<T>>   multi_e;
    std::vector<GridArray<T>>   multi_r;
    std::vector<GridArray<T>>   multi_b;
    std::vector<GridArray<int>> multi_bc;

    Multigrid(int3 N, double3 L, GridArray<int> bct, std::string grid_type = "cell-centered",
              double tol_relative = 1e-9, double tol_descending = 1e-9, int num_presmooth = 2, int num_aftersmooth = 2,
              int num_exactsmooth = 100, int max_iteration = 50, int N_coarsest = 3)
        : _N(N), _L(L), _h({L.x / N.x, L.y / N.y, L.z / N.z}), _grid_type(grid_type), _tol_relative(tol_relative),
          _tol_descending(tol_descending), _num_presmooth(num_presmooth), _num_aftersmooth(num_aftersmooth),
          _num_exactsmooth(num_exactsmooth), _max_iteration(max_iteration), _N_coarsest(N_coarsest), _bct(bct) {
        // 计算网格层数和每一层网格的dh和dim
        auto [num_levels, level_info] = compute_multigrid_levels_3D(N.x, N.y, N.z, L.x, L.y, L.z);
        _level_info                   = level_info;
        _num_levels                   = num_levels;
    }
    void init() {
        LOG_F(WATCH, "init Multigrid.");
        for (const auto& info : _level_info) {
            int3 N_raw = {info.N.x + pad.x, info.N.y + pad.y, info.N.z + pad.z};
            LOG_F(WATCH, "Allocate memory for multigrid solver : %04d %04d %04d %.6e %.6e %.6e", N_raw.x, N_raw.y,
                  N_raw.z, info.dh.x, info.dh.y, info.dh.z);
            multi_x.push_back(create_array<GridArray<T>>(N_raw));
            multi_e.push_back(create_array<GridArray<T>>(N_raw));
            multi_r.push_back(create_array<GridArray<T>>(N_raw));
            multi_b.push_back(create_array<GridArray<T>>(N_raw));
            multi_bc.push_back(create_array<GridArray<int>>(N_raw));
        }
        Kokkos::deep_copy(multi_bc[0], _bct);
        for (int i = 1; i < _num_levels; i++) {
            restrict_bc(multi_bc[i], multi_bc[i - 1], _level_info[i].N);
        }
    }

    void test_restrict(GridArray<T>& r_prime) {
        multi_r[0] = r_prime;
        for (int i = 1; i < _num_levels; i++) {
            restrict(multi_r[i], multi_r[i - 1], _level_info[i].N);
        }
    }

    void test_interpolate() {
        for (int i = 1; i < _num_levels; i++) {
            interpolate(multi_r[i - 1], multi_r[i], _level_info[i].N);
        }
    }

    // NOTE: x are calculated in place.
    void vcycle(GridArray<T>& x, const GridArray<T>& b) {
        // 如果右端项为零，返回 x=0 。
        auto norm_b = norm2(b);
        if (norm_b < NPUHEART_EPS_LARGE) {
            zero(x);
            return;
        }

        // NOTE: copy into data, only shallow copy are needed.
        copy(multi_x[0], x);
        copy(multi_b[0], b);

        // compute residual before calculation
        auto r0 = compute_residual_norm();
        LOG_F(WATCH, "residual0 = %.20e", r0);
        LOG_F(WATCH, "_tol_relative : %.20e", _tol_relative);
        LOG_F(WATCH, "_tol_descending * r0 = %.20e.", _tol_descending * r0);

        for (int i = 0; i < _max_iteration; i++) {
            iterate(x, b, 0);
            auto r = compute_residual_norm();
            LOG_F(WATCH, "iteration : %d, residual : %.20e", i, r);
            // 相对误差小于阈值 或 误差下降量小于阈值
            if (r < _tol_relative * norm_b || r < _tol_descending * r0) { break; }
        }
    }

    double norm2(const GridArray<T>& data) {
        double result = 0.0;
        auto   Nx     = data.extent(0);
        auto   Ny     = data.extent(1);
        auto   Nz     = data.extent(2);

        typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy;
        Kokkos::parallel_reduce(
            "norm2", mdrange_policy({0, 0, 0}, {Nx, Ny, Nz}),
            KOKKOS_LAMBDA(int i, int j, int k, double& result) { result += data(i, j, k) * data(i, j, k); }, result);

        // HACK: use team policy to make parallel_reduce faster.
        // BUG:  TeamThreadMDRange is not working.
        // BUG:  https://github.com/kokkos/kokkos/issues/6530
        //
        // typedef Kokkos::TeamPolicy<>     team_policy;
        // typedef team_policy::member_type member_type;
        // Kokkos::parallel_reduce(
        //     "residual_norm", team_policy(Nx, Kokkos::AUTO),
        //     KOKKOS_LAMBDA(const member_type& teamMember, double& update) {
        //         const int i = teamMember.league_rank();
        //         double local_result = 0;
        //         auto thread_policy = Kokkos::TeamThreadMDRange<Kokkos::Rank<2>,
        //         member_type>(teamMember, Ny, Nz); Kokkos::parallel_reduce(
        //             thread_policy,
        //             [&](const int j, const int k, double& update) {
        //                 // if (teamMember.league_rank() == 0 &&
        //                 teamMember.team_rank() == 0) LOG_F(INFO,"league = %d,
        //                 team = %d, i = %d, j = %d, k = %d",
        //                 teamMember.league_rank(),
        //                        teamMember.team_rank(), i, j, k);
        //                 local_result += residual(i, j, k) * residual(i, j, k);
        //             },
        //             local_result);
        //         // teamMember.team_barrier();

        //         // if (teamMember.team_rank() == 0)
        //         {

        //             // LOG_F(INFO,"league_rank %d, team_rank %d, local_result =
        //             %.20e", i, teamMember.team_rank(),
        //             //        local_result);
        //             Kokkos::single(Kokkos::PerThread(teamMember),
        //             [local_result, &update]() { update += local_result;
        //             });
        //             // update += local_result;
        //             // Kokkos::atomic_add(&update, local_result);
        //         }
        //     },
        //     result);

        return std::sqrt(result * _h.x * _h.y * _h.z);
    }

    double compute_residual_norm() {
        smooth_boundary(multi_x[0], multi_bc[0], _level_info[0].N);
        compute_residual(multi_r[0], multi_x[0], multi_bc[0], multi_b[0], _level_info[0].N, _L);
        return norm2(multi_r[0]);
    }

    void two_grid(GridArray<T>& x, const GridArray<T>& b) {
        auto level = 0;

        copy(multi_x[0], x);
        copy(multi_b[0], b);

        smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
        smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);

        smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
        compute_residual(multi_r[level], multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);

        restrict(multi_b[level + 1], multi_r[level], _level_info[level + 1].N);

        smooth_boundary(multi_x[level + 1], multi_bc[level + 1], _level_info[level + 1].N);
        smooth(multi_x[level + 1], multi_bc[level + 1], multi_b[level + 1], _level_info[level + 1].N, _L);

        interpolate(multi_e[level], multi_x[level + 1], _level_info[level + 1].N);

        add(multi_x[level], multi_e[level]);

        smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
        smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);
    }

    void iterate(GridArray<T>& x, const GridArray<T>& b, int level) {
        if (x.size() != b.size()) { throw std::invalid_argument("WRONG size."); }

        if (level == 0) {
            // NOTE: only shallow copy are needed.
            copy(multi_x[0], x);
            copy(multi_b[0], b);
        }

        for (int iter = 0; iter < _num_presmooth; ++iter) {
            smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
            smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);
        }

        if (level < _num_levels - 1) {
            zero(multi_x[level + 1]);
            smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
            compute_residual(multi_r[level], multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N,
                             _L); // r = b - A(x)
            restrict(multi_b[level + 1], multi_r[level],
                     _level_info[level + 1].N); // b = R(r)
            iterate(multi_x[level + 1], multi_b[level + 1],
                    level + 1); // x = V(x)
            interpolate(multi_e[level], multi_x[level + 1],
                        _level_info[level + 1].N); // e = Ie
            add(multi_x[level], multi_e[level]);   // x = x + e
        } else {
            for (int iter = 0; iter < _num_exactsmooth; ++iter) {
                smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
                smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);
            }
        }

        for (int iter = 0; iter < _num_aftersmooth; ++iter) {
            smooth_boundary(multi_x[level], multi_bc[level], _level_info[level].N);
            smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].N, _L);
        }
    }

    virtual ~Multigrid() {}

    virtual void interpolate(GridArray<T>& fine, const GridArray<T>& coarse, int3 N) {
        CHECK_F(false, "Not implemented yet.");
    }

    virtual void restrict(GridArray<T>& coarse, const GridArray<T>& fine, int3 N) {
        CHECK_F(false, "Not implemented yet.");
    };

    virtual void restrict_bc(GridArray<int>& coarse, const GridArray<int>& fine, int3 N) {
        CHECK_F(false, "Not implemented yet.");
    }

    virtual void smooth(GridArray<T>& p, const GridArray<int>& bct, const GridArray<T>& b, int3 N, double3 L) {
        CHECK_F(false, "Not implemented yet.");
    }

    virtual void smooth_boundary(GridArray<T>& p, const GridArray<int>& bct, int3 N) {
        CHECK_F(false, "Not implemented yet.");
    }

    virtual void compute_residual(GridArray<T>& r, GridArray<T>& p, const GridArray<int>& bct, const GridArray<T>& b,
                                  int3 N, double3 L) {
        CHECK_F(false, "Not implemented yet.");
    }

    void extract_x(Array3D<double>& data_, int level) { extract(data_, multi_x[level], level); }
    void extract_r(Array3D<double>& data_, int level) { extract(data_, multi_r[level], level); }
    void extract_e(Array3D<double>& data_, int level) { extract(data_, multi_e[level], level); }
    void extract_b(Array3D<double>& data_, int level) { extract(data_, multi_b[level], level); }

    void extract(Array3D<double>& data_, GridArray<T> data, int level) {
        auto Nx = data.extent(0);
        auto Ny = data.extent(1);
        auto Nz = data.extent(2);
        CHECK_F(Nx == data_.size() && Ny == data_[0].size() && Nz == data_[0][0].size(), "Size not match!");

        auto data_h = Kokkos::create_mirror_view(data);
        Kokkos::deep_copy(data_h, data);
        for (size_t i = 0; i < Nx; i++)
            for (size_t j = 0; j < Ny; j++)
                for (size_t k = 0; k < Nz; k++) {
                    data_[i][j][k] = data_h(i, j, k);
                }
    }
};

// 通过工厂函数来创建Multigrid的派生类的对象
// 限制类型为Multigrid的派生类
template <typename DerivedMultigrid, typename MultigridBaseTemplate>
concept IsDerivedFromMultigrid = std::is_base_of_v<MultigridBaseTemplate, DerivedMultigrid>;

template <typename T, typename Derived, template <typename> class GridArray>
concept IsDerivedFromMultigridTemplate = IsDerivedFromMultigrid<Derived, Multigrid<T, GridArray>>;

template <typename T, typename Derived, template <typename> class GridArray>
    requires IsDerivedFromMultigridTemplate<T, Derived, GridArray>
class MultigridFactory {
  public:
    static std::unique_ptr<Multigrid<T, GridArray>> createObject(int3 N, double3 L, GridArray<int> bct) {
        // 使用 std::make_unique 代替 new
        auto obj = std::make_unique<Derived>(N, L, bct);
        obj->init();
        return obj;
    }
};
} // namespace mg
