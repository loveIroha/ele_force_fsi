#pragma once
#include <dolfin.h>
#include <mpi.h>
#include "GPB_cell_model.h"
// #include "ActiveContractionGPB.h"  // 已移除，NHS 模型不再使用
#include "SDIRK3_Solver.h"
#include <vector>
#include <memory>
#include <omp.h>
#include <cmath>
#include <algorithm>

// ===========================================================================
// 修改说明（问题3 + 问题4）：
//
//   问题3：NHS 模型传入固定 lmbda=1.0, dlambda=0.0，完全忽略力学反馈
//   问题4：主动力模型改为 Land (2017) 横桥模型，以 XS/XW 驱动 UFL 中的 Ta
//
//   旧方案（NHS）：
//     update_nhs_state(state, Ca_i, 1.0, 0.0, dt)  // lmbda 固定
//     → get_active_tension() → 设置 UFL 中的标量 T
//
//   新方案（Land）：
//     update_land_ode(state, Ca_i, lmbda, Zetas, Zetaw, dt)
//       lmbda/Zetas/Zetaw 来自上一力学步（update_mechanics_feedback 提供）
//     → get_XS_vector(), get_XW_vector() → 设置 UFL 中的 XS, XW 系数
//
//   电生理部分（GPBCellModel + SDIRK3 + subcycling）完全未修改
// ===========================================================================

// ---------------------------------------------------------------------------
// Land 模型参数（与 UFL 保持完全一致，单位 ms⁻¹ 或无量纲）
// ---------------------------------------------------------------------------
namespace LandParams {
    constexpr double ktrpn     = 0.1;    // TnC 结合率 (ms⁻¹)
    constexpr double ku        = 0.04;   // 横桥自发解离率 (ms⁻¹)
    constexpr double kuw       = 0.182;  // U→W 转换率 (ms⁻¹)
    constexpr double kws       = 0.012;  // W→S 转换率 (ms⁻¹)
    constexpr double rs        = 0.25;   // 强结合横桥平衡比例
    constexpr double rw        = 0.5;    // 弱结合横桥平衡比例
    constexpr double phi       = 2.23;   // 横桥功率冲程参数
    // 推导速率常数
    constexpr double kwu = -kws + kuw * (1.0/rw - 1.0);  // ≈ 0.170 ms⁻¹
    constexpr double ksu = kws * rw * (1.0/rs - 1.0);    // ≈ 0.018 ms⁻¹
    // CaTrpn 参数
    constexpr double ntrpn     = 2.0;
    constexpr double ntm       = 2.4;
    constexpr double Trpn50    = 0.35;
    constexpr double cat50_ref = 0.535;  // μM
    constexpr double Beta1     = -2.4;   // μM per unit stretch
    // 应变相关脱离系数
    constexpr double gammas    = 0.0085; // ms⁻¹
    constexpr double gammaw    = 0.615;  // ms⁻¹
}

// ---------------------------------------------------------------------------
// Land 模型状态（替换 NHSState）
// ---------------------------------------------------------------------------
struct LandState {
    double XS     = 0.0;  // 强结合横桥比例
    double XW     = 0.0;  // 弱结合横桥比例
    double CaTrpn = 0.0;  // 与 TnC 结合的 Ca 比例
    double TmB    = 1.0;  // 被原肌球蛋白阻断的位点比例（初始=1，全阻断）
};

// ===========================================================================
// GPBTissueManager
// ===========================================================================
class GPBTissueManager
{
public:
    GPBTissueManager(
        std::shared_ptr<dolfin::FunctionSpace> function_space,
        double dt_pde_ms,
        double dt_ode_ms)
        : V_space(function_space),
          dt_pde_milliseconds(dt_pde_ms),
          dt_ode_milliseconds(dt_ode_ms)
    {
        n_substeps = static_cast<int>(std::round(dt_pde_ms / dt_ode_ms));
        if (n_substeps < 1) {
            std::cerr << "Warning: dt_ODE > dt_PDE, setting n_substeps = 1" << std::endl;
            n_substeps = 1;
            dt_ode_milliseconds = dt_pde_milliseconds;
        }
        dt_ode_milliseconds = dt_pde_milliseconds / n_substeps;

        auto dof_range   = V_space->dofmap()->ownership_range();
        num_nodes_local  = dof_range.second - dof_range.first;
        num_nodes_global = V_space->dim();

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "初始化 GPB 组织管理器 (多尺度时间步进)" << std::endl;
        std::cout << "  本地节点数: " << num_nodes_local
                  << " / 全局: " << num_nodes_global << std::endl;
        std::cout << "  PDE 时间步长: " << dt_pde_milliseconds << " ms" << std::endl;
        std::cout << "  ODE 时间步长: " << dt_ode_milliseconds << " ms" << std::endl;
        std::cout << "  子循环步数:   " << n_substeps << std::endl;
        std::cout << "  主动收缩模型: Land (2017) 横桥模型" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;

        // GPB 电生理状态（未修改）
        cell_states.resize(num_nodes_local);
        std::vector<double> resting = GPBCellModel::get_initial_state();
        for (size_t i = 0; i < num_nodes_local; ++i)
            cell_states[i] = resting;

        // Land 状态（替换 nhs_states）
        land_states.resize(num_nodes_local);   // LandState 默认构造：XS=0,XW=0,CaTrpn=0,TmB=1

        // 力学反馈（替换 NHS 中固定值 1.0/0.0）
        lmbda_nodes.assign(num_nodes_local, 1.0);
        Zetas_nodes.assign(num_nodes_local, 0.0);
        Zetaw_nodes.assign(num_nodes_local, 0.0);

        std::cout << "  所有细胞已初始化为静息态" << std::endl;
        std::cout << "  Land 横桥模型启用 (XS/XW → UFL 系数)" << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 电生理 ODE 子循环（未修改）+ Land ODE 同步更新（替换 NHS 调用）
    // -----------------------------------------------------------------------
    void compute_ionic_current_subcycling(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms)
    {
        std::vector<double> Vm_array;
        Vm_function->vector()->get_local(Vm_array);

        if (Vm_array.size() != num_nodes_local)
            throw std::runtime_error("Vm array size mismatch! Expected: "
                + std::to_string(num_nodes_local) + ", Got: "
                + std::to_string(Vm_array.size()));

        std::vector<double> I_ion_avg(num_nodes_local, 0.0);

        for (int substep = 0; substep < n_substeps; ++substep) {
            double t_sub = time_ms + substep * dt_ode_milliseconds;

            #pragma omp parallel for schedule(static)
            for (size_t node = 0; node < num_nodes_local; ++node) {

                // ── GPB 电生理 ODE（SDIRK3，完全未修改）──
                double Vm_mV = Vm_array[node] * 1000.0;

                auto rhs_func = [Vm_mV](double t, std::vector<double>& y, double /*Vm_fixed*/) {
                    return GPBCellModel::compute_ionic_current(t, y, Vm_mV);
                };

                typename SDIRK3Solver<std::vector<double>, GPBCellModel::Result>::Config cfg;
                cfg.max_iter     = 3;
                cfg.adaptive_iter = true;
                cfg.tol          = 1e-6;

                double I_ion = SDIRK3Solver<std::vector<double>, GPBCellModel::Result>::step(
                    rhs_func, t_sub, cell_states[node], Vm_mV,
                    dt_ode_milliseconds, cfg);

                if (std::isnan(I_ion) || std::isinf(I_ion)) I_ion = 0.0;
                I_ion_avg[node] += I_ion;

                // ── Land ODE（替换 NHS 调用，问题3+4 修复）──
                // GPB cell_states[37] 是胞内 Ca²⁺，单位 mM → 转换为 μM
                double Ca_i_uM = cell_states[node][37] * 1000.0;

                // 使用力学步传入的真实 lmbda/Zetas/Zetaw（不再固定为 1/0）
                update_land_ode(
                    land_states[node],
                    Ca_i_uM,
                    lmbda_nodes[node],
                    Zetas_nodes[node],
                    Zetaw_nodes[node],
                    dt_ode_milliseconds);
            }
        }

        #pragma omp parallel for simd
        for (size_t node = 0; node < num_nodes_local; ++node)
            I_ion_avg[node] /= n_substeps;

        I_ion_function->vector()->set_local(I_ion_avg);
        I_ion_function->vector()->apply("insert");
    }

    // 单步版本（兼容接口）
    void compute_ionic_current(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double time_ms)
    {
        compute_ionic_current_subcycling(Vm_function, I_ion_function, time_ms);
    }

    // -----------------------------------------------------------------------
    // 力学步收敛后由 main.cpp 调用（问题3修复核心）
    //
    // 传入当前步收敛后的 lmbda, Zetas, Zetaw 节点值，
    // 供下一 EP 子循环中 Land ODE 的 cat50 和应变脱离率计算使用
    // -----------------------------------------------------------------------
    void update_mechanics_feedback(
        const std::vector<double>& lmbda_arr,
        const std::vector<double>& Zetas_arr,
        const std::vector<double>& Zetaw_arr)
    {
        if (lmbda_arr.size() != num_nodes_local) {
            std::cerr << "Warning: update_mechanics_feedback size mismatch ("
                      << lmbda_arr.size() << " vs " << num_nodes_local
                      << "), skipping" << std::endl;
            return;
        }
        lmbda_nodes = lmbda_arr;
        Zetas_nodes = Zetas_arr;
        Zetaw_nodes = Zetaw_arr;
    }

    // -----------------------------------------------------------------------
    // XS/XW 输出（供 main.cpp 设置 UFL 系数，替换旧的 get_active_tension）
    // -----------------------------------------------------------------------
    std::vector<double> get_XS_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = land_states[i].XS;
        return out;
    }

    std::vector<double> get_XW_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = land_states[i].XW;
        return out;
    }

    // Land 主动收缩力（与固体 UFL 中 Ta 一致的节点表达）
    std::vector<double> get_active_tension_vector() const {
        std::vector<double> out(num_nodes_local, 0.0);
        for (size_t i = 0; i < num_nodes_local; ++i) {
            const double lmbda_c  = std::min(lmbda_nodes[i], 1.2);
            const double h_prima  = 1.0 + 2.3 * (lmbda_c + std::min(lmbda_c, 0.87) - 1.87);
            const double h_lambda = std::max(0.0, h_prima);
            out[i] = h_lambda * (8.4e5 / 0.25) *
                     (land_states[i].XS * (Zetas_nodes[i] + 1.0) + land_states[i].XW * Zetaw_nodes[i]);
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // 统计接口（与旧版本接口兼容）
    // -----------------------------------------------------------------------
    double get_dt_pde_ms()  const { return dt_pde_milliseconds; }
    double get_dt_ode_ms()  const { return dt_ode_milliseconds; }
    int    get_n_substeps() const { return n_substeps; }

    std::vector<double> get_Ca_i_vector() const {
        std::vector<double> out(num_nodes_local);
        for (size_t i = 0; i < num_nodes_local; ++i)
            out[i] = cell_states[i][37] * 1000.0;  // mM → μM
        return out;
    }

    struct Statistics {
        double Vm_min_mV, Vm_max_mV, Vm_mean_mV;
        double I_ion_min, I_ion_max, I_ion_mean;
        double Ca_i_min_mM, Ca_i_max_mM, Ca_i_mean_mM;
        size_t num_nodes;
    };

    Statistics get_statistics(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function) const
    {
        Statistics s;
        s.num_nodes  = num_nodes_global;
        s.Vm_min_mV  = Vm_function->vector()->min()  * 1000.0;
        s.Vm_max_mV  = Vm_function->vector()->max()  * 1000.0;
        s.Vm_mean_mV = Vm_function->vector()->sum()  / num_nodes_global * 1000.0;
        s.I_ion_min  = I_ion_function->vector()->min();
        s.I_ion_max  = I_ion_function->vector()->max();
        s.I_ion_mean = I_ion_function->vector()->sum() / num_nodes_global;

        double ca_min = 1e9, ca_max = -1e9, ca_sum = 0.0;
        for (size_t i = 0; i < num_nodes_local; ++i) {
            double ca = cell_states[i][37] * 1000.0;
            if (ca < ca_min) ca_min = ca;
            if (ca > ca_max) ca_max = ca;
            ca_sum += ca;
        }
        double gmin, gmax, gsum;
        MPI_Allreduce(&ca_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&ca_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&ca_sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        s.Ca_i_min_mM  = gmin;
        s.Ca_i_max_mM  = gmax;
        s.Ca_i_mean_mM = gsum / num_nodes_global;
        return s;
    }

    // Land 统计（替换旧 TensionStats）
    struct LandStats {
        double XS_min, XS_max, XS_mean;
        double XW_min, XW_max, XW_mean;
        double CaTrpn_mean;
        double lmbda_mean;
    };
    using TensionStats = LandStats;

    LandStats get_land_statistics() const {
        double xs_min=1e9, xs_max=-1e9, xs_sum=0.0;
        double xw_min=1e9, xw_max=-1e9, xw_sum=0.0;
        double ct_sum=0.0, lm_sum=0.0;
        for (size_t i = 0; i < num_nodes_local; ++i) {
            double xs = land_states[i].XS, xw = land_states[i].XW;
            if (xs < xs_min) xs_min = xs;  if (xs > xs_max) xs_max = xs;  xs_sum += xs;
            if (xw < xw_min) xw_min = xw;  if (xw > xw_max) xw_max = xw;  xw_sum += xw;
            ct_sum += land_states[i].CaTrpn;
            lm_sum += lmbda_nodes[i];
        }
        double gxs_min, gxs_max, gxs_sum;
        double gxw_min, gxw_max, gxw_sum;
        double gct_sum, glm_sum;
        MPI_Allreduce(&xs_min, &gxs_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&xs_max, &gxs_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&xs_sum, &gxs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_min, &gxw_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_max, &gxw_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&xw_sum, &gxw_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&ct_sum, &gct_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&lm_sum, &glm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        LandStats s;
        s.XS_min = gxs_min;  s.XS_max = gxs_max;  s.XS_mean = gxs_sum / num_nodes_global;
        s.XW_min = gxw_min;  s.XW_max = gxw_max;  s.XW_mean = gxw_sum / num_nodes_global;
        s.CaTrpn_mean = gct_sum / num_nodes_global;
        s.lmbda_mean  = glm_sum / num_nodes_global;
        return s;
    }

    TensionStats get_tension_statistics() const {
        return get_land_statistics();
    }

    bool check_stability(
        std::shared_ptr<dolfin::Function> Vm_function,
        std::shared_ptr<dolfin::Function> I_ion_function,
        double acceptable_Vm_range_mV = 200.0) const
    {
        double vmin = Vm_function->vector()->min() * 1000.0;
        double vmax = Vm_function->vector()->max() * 1000.0;
        if (std::isnan(vmin)||std::isnan(vmax)||std::isinf(vmin)||std::isinf(vmax)) {
            std::cerr << "Error: Vm contains NaN/Inf!" << std::endl; return false;
        }
        if (vmax > acceptable_Vm_range_mV || vmin < -acceptable_Vm_range_mV) {
            std::cerr << "Warning: Vm out of range [" << vmin << ", " << vmax << "] mV\n";
            return false;
        }
        double imin = I_ion_function->vector()->min();
        double imax = I_ion_function->vector()->max();
        if (std::isnan(imin)||std::isnan(imax)||std::isinf(imin)||std::isinf(imax)) {
            std::cerr << "Error: I_ion contains NaN/Inf!" << std::endl; return false;
        }
        return true;
    }

    const std::vector<double>& get_cell_state(size_t idx) const {
        if (idx >= num_nodes_local) throw std::out_of_range("Node index out of range");
        return cell_states[idx];
    }

    void print_debug_info(size_t node = 0) const {
        if (node >= num_nodes_local) return;
        std::cout << "\nNode " << node << ":" << std::endl;
        std::cout << "  Ca_i    = " << cell_states[node][37]*1000.0 << " uM\n"
                  << "  XS      = " << land_states[node].XS
                  << "  XW      = " << land_states[node].XW
                  << "  CaTrpn  = " << land_states[node].CaTrpn << "\n"
                  << "  lmbda   = " << lmbda_nodes[node]
                  << "  Zetas   = " << Zetas_nodes[node]
                  << "  Zetaw   = " << Zetaw_nodes[node] << std::endl;
    }

private:
    // -----------------------------------------------------------------------
    // Land 模型 ODE 单步更新（前向欧拉）
    //
    // 状态方程（与 UFL 中 UFL 形式定义完全一致）：
    //   dCaTrpn/dt = ktrpn*( (Ca/cat50)^ntrpn*(1-CaTrpn) - CaTrpn )
    //   dTmB/dt    = kb*CaTrpn^(-ntm/2)*XU - ku*CaTrpn^(ntm/2)*TmB
    //   dXW/dt     = kuw*XU - kws*XW - XW*gammawu - XW*kwu
    //   dXS/dt     = kws*XW - XS*gammasu - XS*ksu
    // -----------------------------------------------------------------------
    static void update_land_ode(
        LandState& s,
        double Ca_i_uM,
        double lmbda,
        double Zetas,
        double Zetaw,
        double dt_ms)
    {
        using namespace LandParams;

        // Frank-Starling：lmbda 影响 Ca₅₀（与 UFL 的 cat50 表达式一致）
        double lmbda_c = (lmbda < 1.2) ? lmbda : 1.2;
        double cat50   = cat50_ref + Beta1 * (lmbda_c - 1.0);
        if (cat50 < 0.1) cat50 = 0.1;

        // 应变相关横桥脱离率（与 UFL 的 gammasu/gammawu 定义一致）
        double zetas1  = (Zetas >  0.0) ?  Zetas         : 0.0;
        double zetas2  = (Zetas < -1.0) ? (-1.0 - Zetas) : 0.0;
        double gammasu = gammas * ((zetas1 > zetas2) ? zetas1 : zetas2);
        double gammawu = gammaw * std::fabs(Zetaw);

        // 状态变量夹紧（防止数值负值）
        double XS_c     = (s.XS     > 0.0) ? s.XS     : 0.0;
        double XW_c     = (s.XW     > 0.0) ? s.XW     : 0.0;
        double CaTrpn_c = (s.CaTrpn > 0.0) ? s.CaTrpn : 0.0;
        double TmB_c    = (s.TmB    > 0.0) ? s.TmB    : 0.0;
        double XU = 1.0 - TmB_c - XS_c - XW_c;
        if (XU < 0.0) XU = 0.0;

        // 幂次项（防止溢出）
        double kb = ku * std::pow(Trpn50, ntm) / (1.0 - rs - rw*(1.0-rs));

        double Ca_ratio = Ca_i_uM / cat50;
        double CaTrpn_act   = std::pow(Ca_ratio, ntrpn);                          // (Ca/cat50)^ntrpn
        double CaTrpn_phalf = (CaTrpn_c > 1e-12) ? std::pow(CaTrpn_c,  ntm*0.5) : 0.0;
        double CaTrpn_nhalf = (CaTrpn_c > 1e-12)
            ? std::min(std::pow(CaTrpn_c, -ntm*0.5), 100.0) : 100.0;

        // ODE 右端
        double dCaTrpn = ktrpn * (-CaTrpn_c + CaTrpn_act * (1.0 - CaTrpn_c));
        double dTmB    = kb  * CaTrpn_nhalf * XU  - ku  * CaTrpn_phalf * TmB_c;
        double dXW     = kuw * XU - kws * XW_c - XW_c * gammawu - XW_c * kwu;
        double dXS     = kws * XW_c - XS_c * gammasu - XS_c * ksu;

        // 前向欧拉更新
        s.CaTrpn += dt_ms * dCaTrpn;
        s.TmB    += dt_ms * dTmB;
        s.XW     += dt_ms * dXW;
        s.XS     += dt_ms * dXS;

        // 物理约束夹紧
        if (s.CaTrpn < 0.0) s.CaTrpn = 0.0;
        if (s.XW     < 0.0) s.XW     = 0.0;
        if (s.XS     < 0.0) s.XS     = 0.0;
        if (s.TmB    < 0.0) s.TmB    = 0.0;
        if (s.TmB    > 1.0) s.TmB    = 1.0;
    }

    // 成员变量
    std::shared_ptr<dolfin::FunctionSpace> V_space;
    size_t num_nodes_local;
    size_t num_nodes_global;
    double dt_pde_milliseconds;
    double dt_ode_milliseconds;
    int    n_substeps;

    // 电生理状态（未修改）
    std::vector<std::vector<double>> cell_states;

    // Land 模型状态（替换 nhs_states + active_tension）
    std::vector<LandState> land_states;

    // 力学反馈（替换 NHS 固定值：lmbda=1, dlambda=0）
    std::vector<double> lmbda_nodes;
    std::vector<double> Zetas_nodes;
    std::vector<double> Zetaw_nodes;
};
