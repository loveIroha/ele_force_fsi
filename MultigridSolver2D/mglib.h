/// @date 2023-08-12
/// @file mglib.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief function for multigrid solver
///
///

#ifndef __MGLIB_H__
#define __MGLIB_H__

#include <io/vector_io.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace mg {
// TODO: 删除这两个变量
const int NEUMANN   = 2;
const int DIRICHLET = 1;

template <typename T>
using vector2D = std::vector<std::vector<T>>;

struct LevelInfo {
    double dx;
    double dy;
    int    Nx;
    int    Ny;
};

std::tuple<int, std::vector<LevelInfo>> compute_multigrid_levels_2D(int Nx, int Ny, double width, double height);

vector2D<double> interpolate(const vector2D<double>& coarse, int Nx, int Ny);
vector2D<double> restrict(const vector2D<double>& fine, int Nx, int Ny);
vector2D<int> restrict_bc(const vector2D<int>& fine, int Nx, int Ny);

std::vector<std::vector<double>> compute_residual(std::vector<std::vector<double>>&       phi,
                                                  const std::vector<std::vector<double>>& b, int Nx, int Ny,
                                                  double width, double height);

std::vector<std::vector<double>> generate_u_prime(const std::vector<std::vector<int>>&    boundary_type,
                                                  const std::vector<std::vector<double>>& boundary_values, int Nx,
                                                  int Ny, double width, double height);

std::vector<std::vector<double>> smooth(std::vector<std::vector<double>>&       phi,
                                        const std::vector<std::vector<int>>&    boundary_type,
                                        const std::vector<std::vector<double>>& b, int Nx, int Ny, double width,
                                        double height);
std::vector<std::vector<double>> compute_residual(std::vector<std::vector<double>>&       phi,
                                                  const std::vector<std::vector<int>>&    boundary_type,
                                                  const std::vector<std::vector<double>>& b, int Nx, int Ny,
                                                  double width, double height);
} // namespace mg
#endif
