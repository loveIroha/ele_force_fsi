/// @date 2024-04-20
/// @file TubeFlow.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#pragma once

#include "Kokkos/StokesFlow3D_kokkos.h"

namespace {


// 辅助处理从文本读入的入口压强数据

double linearInterpolation(double x0, double y0, double x1, double y1, double x) {
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

double getInterpolatedValue(const std::vector<double>& data, double currentTime, double startTime,
                            double timeInterval) {
    int index = static_cast<int>((currentTime - startTime) / timeInterval);
    printf("index %d currentTime %f startTime %f timeInterval %f\n", index, currentTime, startTime, timeInterval);
    double t0 = startTime + index * timeInterval;
    double t1 = t0 + timeInterval;
    if (index >= data.size() - 1) return data.back();
    if (index < 0) return data.front();
    return linearInterpolation(t0, data[index], t1, data[index + 1], currentTime);
}

inline double delta_4(const double& r) {
    const double r1 = abs(r);
    const double r2 = r * r;
    if (r1 < 1.0) {
        return 0.125 * (3.0 - 2.0 * r1 + sqrt(1.0 + 4.0 * r1 - 4.0 * r2));
    } else if (r1 < 2.0) {
        return 0.125 * (5.0 - 2.0 * r1 - sqrt(-7.0 + 12.0 * r1 - 4.0 * r2));
    } else {
        return 0.0;
    }
} // delta_4
} // namespace

namespace stokes_flow {
template <int DIM>
class TubeFlow : public StokesFlow<DIM> {
  public:
    std::vector<double> input_pressure;           // 输入的压力数据
    double              dt_input_pressuret;       // 压力数据的时间间隔
    double              t0_input_pressuret = 0.0; // 压力数据的初始时间
    double              radius             = 3.0; // 圆心为(5.0, 5.0, z)，半径为3.0
    double              center_x           = 5.0;
    double              center_y           = 5.0;
    double              rho                = 1.0; // 流体密度

    TubeFlow(std::shared_ptr<NavierStokesDemo> _ns_demo, std::string _output_file)
        : StokesFlow<DIM>(_ns_demo, _output_file + "fluid/data.pvd") {
        // 设置边界类型和边界值
        std::array<int, 6> all_boundary_type   = {1, 1, 1, 1, 2, 2};
        std::array<int, 6> all_boundary_type_p = {2, 2, 2, 2, 1, 1};
        StokesFlow<DIM>::all_boundary_type_p   = all_boundary_type_p;
        StokesFlow<DIM>::all_boundary_type     = all_boundary_type;

        // NOTE: readtxt_1 是为高老师给的txt定制的数据读取函数，返回输入数据的时间间隔
        
        auto pressure_file = geometry_path("MV-Gao/other/input_pressure.txt");
        
        dt_input_pressuret = readtxt_1(pressure_file, input_pressure);

        // 其他初始化操作
        StokesFlow<DIM>::initial_boundary_conditions();
        set_boundary_conditions();
    }

  private:
    /// @brief 设置初始值
    virtual void initial_values() final override {}

    /// @brief 设置边界值(可能会随时间变化)
    virtual void set_boundary_conditions() final override {
        double dx = StokesFlow<DIM>::_dx;
        double dy = StokesFlow<DIM>::_dy;
        // NOTE: 这里需要加一个负号。因为出口处压强设为零，入口处的压强为读入数据的相反数。
        double current_pressure
            = -getInterpolatedValue(input_pressure, StokesFlow<DIM>::_t, t0_input_pressuret, dt_input_pressuret)
              * ISUnits::mmHg;
        // double current_pressure = StokesFlow<DIM>::_t * 150;
        printf("current_pressure %f\n", current_pressure);
        for (int i = 0; i < StokesFlow<DIM>::_Nx; ++i) {
            for (int j = 0; j < StokesFlow<DIM>::_Ny; ++j) {
                double x             = (i + 0.5) * dx - center_x;
                double y             = (j + 0.5) * dy - center_y;
                double r             = std::sqrt(x * x + y * y);
                double h             = std::max(dx, dy);
                double radial_grader = 1.0;
                // double radial_grader = 1.0 - delta_4((r - radius) / h) / delta_4(0.0);
                // FIXME : 确定此处是否和速度一样需要使用 delta 函数
                if (r < radius) {
                    StokesFlow<DIM>::pbv[i + 1][j + 1][0]                        = 0.0;
                    StokesFlow<DIM>::pbv[i + 1][j + 1][StokesFlow<DIM>::_Nz + 1] = radial_grader * current_pressure;
                } else {
                    StokesFlow<DIM>::pbv[i + 1][j + 1][0]                        = 0.0;
                    StokesFlow<DIM>::pbv[i + 1][j + 1][StokesFlow<DIM>::_Nz + 1] = 0.0;
                }
            }
        }
    }

    /// @brief 设置源项(可能会随时间变化)
    virtual void set_source_terms() final override {
        double dx = StokesFlow<DIM>::_dx;
        double dy = StokesFlow<DIM>::_dy;
        double dz = StokesFlow<DIM>::_dz;
        int    Nx = StokesFlow<DIM>::_Nx;
        int    Ny = StokesFlow<DIM>::_Ny;
        int    Nz = StokesFlow<DIM>::_Nz;
        std::vector<int> z_edges = {1, Nz+1};

        // Z方向上边界速度的w分量
        for (auto z_edge : z_edges) 
        for (int i = 0; i < Nx; ++i) { 
            for (int j = 0; j < Ny; ++j) {
                double x       = (i + 0.5) * dx - center_x;
                double y       = (j + 0.5) * dy - center_y;
                double r       = std::sqrt(x * x + y * y);
                double d_kappa = 0.5 * rho / StokesFlow<DIM>::_dt;
                // TODO : 高昊在此处用了 delta 函数，目的是直管壁面的速度为零，靠近直管壁面的速度是连续变化的
                double radial_grader = 0.0;
                if (r > radius) {
                    radial_grader = 1.0;
                } else if (StokesFlow<DIM>::_t > 0.185 && StokesFlow<DIM>::_t < 0.6) {
                    // 瓣膜闭合的时间段，直管内部区域，速度大于零，施加惩罚项力
                    if (StokesFlow<DIM>::wn[i + 1][j + 1][z_edge] > 0) { radial_grader = 1.0; }
                } else {
                    // 瓣膜打开的时间段，直管内部区域，不施加惩罚项目
                    radial_grader = 0.0;
                }
                StokesFlow<DIM>::f3[i + 1][j + 1][z_edge] -= radial_grader * d_kappa * StokesFlow<DIM>::wn[i + 1][j + 1][z_edge];
            }
        }
        printf("In set_source_terms(): t %f\n", StokesFlow<DIM>::_t);
        // 读取源项数据
        // 读取速度数据
        // 获取坐标数据
        // 利用源项对速度进行惩罚
    }

    /// @brief 后处理
    virtual void post_process(int i) final override {}
};

} // namespace stokes_flow
