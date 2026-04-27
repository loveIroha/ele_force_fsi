/// @date 2023-08-12
/// @file MultigridSolver2D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///

#ifndef __MULTIGRID_SOLVER_H__
#define __MULTIGRID_SOLVER_H__

#include <AlgebraSolver/algebra.h>

#include <vector>

#include "mglib.h"

namespace mg {

class MultigridSolver2D {
  public:
    int         _Nx;
    int         _Ny;
    double      _width;
    double      _height;
    std::string _grid_type;
    double      _tol_relative;
    double      _tol_descending;
    int         _num_presmooth;
    int         _num_aftersmooth;
    int         _num_exactsmooth;
    int         _max_iteration;
    int         _N_coarsest;

    // 计算出来的数据
    double                 _dx;
    double                 _dy;
    std::vector<LevelInfo> _level_info;
    int                    _num_levels;

    std::vector<vector2D<double>> multi_x;
    std::vector<vector2D<double>> multi_e;
    std::vector<vector2D<double>> multi_r;
    std::vector<vector2D<double>> multi_b;
    std::vector<vector2D<int>>    multi_bc;

    MultigridSolver2D(int Nx, int Ny, double width, double height, vector2D<int> boundary_type,
                      std::string grid_type = "cell-centered", double tol_relative = 1e-6, double tol_descending = 1e-6,
                      int num_presmooth = 2, int num_aftersmooth = 2, int num_exactsmooth = 100, int max_iteration = 2,
                      int N_coarsest = 3)
        : _Nx(Nx), _Ny(Ny), _width(width), _height(height), _grid_type(grid_type), _tol_relative(tol_relative),
          _tol_descending(tol_descending), _num_presmooth(num_presmooth), _num_aftersmooth(num_aftersmooth),
          _num_exactsmooth(num_exactsmooth), _max_iteration(max_iteration), _N_coarsest(N_coarsest) {
        _dx = width / Nx;
        _dy = height / Ny;

        // 计算网格层数和每一层网格的dh和dim
        auto [num_levels, level_info] = compute_multigrid_levels_2D(Nx, Ny, width, height);
        _level_info                   = level_info;
        _num_levels                   = num_levels;

        // 分配每一层网格需使用的变量
        for (const auto& info : level_info) {
            std::cout << info.dx << " " << info.dy << " " << info.Nx << " " << info.Ny << std::endl;
            multi_x.push_back(IO::make_vector_2D<double>(info.Nx + 2, info.Ny + 2));
            multi_e.push_back(IO::make_vector_2D<double>(info.Nx + 2, info.Ny + 2));
            multi_r.push_back(IO::make_vector_2D<double>(info.Nx + 2, info.Ny + 2));
            multi_b.push_back(IO::make_vector_2D<double>(info.Nx + 2, info.Ny + 2));
        }

        // 计算每一层网格的边界类型
        multi_bc.push_back(boundary_type);
        for (int i = 1; i < _num_levels; i++) {
            multi_bc.push_back(restrict_bc(multi_bc[i - 1], level_info[i].Nx, level_info[i].Ny));
        }
    }
    // FIXME: x 不应该被改变
    std::vector<std::vector<double>> vcycle(std::vector<std::vector<double>>&       x,
                                            const std::vector<std::vector<double>>& b) {
        // 如果右端项为零，返回 x=0 。
        double norm_b = algebra::norm(b) * std::sqrt(_dx * _dy);
        double eps    = std::numeric_limits<double>::epsilon();
        if (norm_b < eps) {
            algebra::zero(x);
            return x;
        }

        // compute residual before calculation
        multi_r[0]        = compute_residual(x, multi_bc[0], b, _Nx, _Ny, _width, _height);
        double _residual  = algebra::norm(multi_r[0]) * std::sqrt(_dx * _dy);
        double _residual0 = _residual;
        printf("residual0 = %.12e\n", _residual);

        std::vector<std::vector<double>> xn = x;

        for (int i = 0; i < _max_iteration; i++) {
            xn        = iterate(x, b, 0);
            _residual = algebra::norm(multi_r[0]) * std::sqrt(_dx * _dy);
            printf("residual  = %.12e\n", _residual);

            // 相对误差小于阈值 或 误差下降量小于阈值
            if (_residual < _tol_relative * norm_b || _residual < _tol_descending * _residual0) { break; }

            x = xn;
        }

        return xn;
    }

    std::vector<std::vector<double>> iterate(std::vector<std::vector<double>>&       x,
                                             const std::vector<std::vector<double>>& b, int level) {
        if (x.size() != b.size()) { throw std::invalid_argument("WRONG size."); }

        if (level == 0) {
            multi_x[0] = x;
            multi_b[0] = b;
        }

        for (int iter = 0; iter < _num_presmooth; ++iter) {
            multi_x[level] = smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].Nx,
                                    _level_info[level].Ny, _width, _height);
        }

        if (level < _num_levels - 1) {
            multi_x[level + 1].assign(_level_info[level + 1].Nx + 2,
                                      std::vector<double>(_level_info[level + 1].Ny + 2, 0.0));

            multi_r[level] = compute_residual(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].Nx,
                                              _level_info[level].Ny, _width, _height);

            multi_b[level + 1] = restrict(multi_r[level], _level_info[level + 1].Nx, _level_info[level + 1].Ny);

            multi_x[level + 1] = iterate(multi_x[level + 1], multi_b[level + 1], level + 1);

            multi_e[level] = interpolate(multi_x[level + 1], _level_info[level + 1].Nx, _level_info[level + 1].Ny);

            for (int i = 0; i < _level_info[level].Nx + 2; ++i) {
                for (int j = 0; j < _level_info[level].Ny + 2; ++j) {
                    multi_x[level][i][j] += multi_e[level][i][j];
                }
            }
        } else {
            for (int iter = 0; iter < _num_exactsmooth; ++iter) {
                multi_x[level] = smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].Nx,
                                        _level_info[level].Ny, _width, _height);
            }
        }

        for (int iter = 0; iter < _num_aftersmooth; ++iter) {
            multi_x[level] = smooth(multi_x[level], multi_bc[level], multi_b[level], _level_info[level].Nx,
                                    _level_info[level].Ny, _width, _height);
        }

        return multi_x[level];
    }
    // 其他成员函数
    ~MultigridSolver2D() {}
};
} // namespace mg
#endif