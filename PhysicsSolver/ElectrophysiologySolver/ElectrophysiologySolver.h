#ifndef __ELECTROPHYSIOLOGY_SOLVER_H__
#define __ELECTROPHYSIOLOGY_SOLVER_H__

#include <dolfin.h>
#include <io/include/io/loguru.hpp>
#include "GPB_cell_model.h"
#include "GPBTissueManager_Land.h"
#include "Monodomain.h"  // FFC 生成
#include <unordered_set>

namespace dolfin {

/**
 * 纤维方向 Expression 类
 */
class FiberDirectionsEP : public Expression
{
public:
    FiberDirectionsEP(
        std::shared_ptr<MeshFunction<double>> _c0,
        std::shared_ptr<MeshFunction<double>> _c1,
        std::shared_ptr<MeshFunction<double>> _c2) 
        : Expression(3), c0(_c0), c1(_c1), c2(_c2) {}

    void eval(Eigen::Ref<Eigen::VectorXd> values, 
              Eigen::Ref<const Eigen::VectorXd> x, 
              const ufc::cell &cell) const override
    {
        const uint cell_index = cell.index;
        values[0] = (*c0)[cell_index];
        values[1] = (*c1)[cell_index];
        values[2] = (*c2)[cell_index];
    }

    std::shared_ptr<MeshFunction<double>> c0;
    std::shared_ptr<MeshFunction<double>> c1;
    std::shared_ptr<MeshFunction<double>> c2;
};

class StimulusCurrentEP : public Expression
{
public:
    StimulusCurrentEP(double amplitude, double t_start_ms, double t_end_ms,
                      std::shared_ptr<const Mesh> mesh,
                      std::shared_ptr<const MeshFunction<std::size_t>> facet_markers,
                      std::size_t endo_marker)
        : Expression(), amp(amplitude), t_start(t_start_ms), t_end(t_end_ms),
          current_time(0.0), mesh(mesh), facet_markers(facet_markers),
          endo_marker(endo_marker), use_boundary_marker(true)
    {
        build_cell_to_boundary_map();
    }

    StimulusCurrentEP(double amplitude, double t_start_ms, double t_end_ms)
        : Expression(), amp(amplitude), t_start(t_start_ms), t_end(t_end_ms), 
          current_time(0.0), mesh(nullptr), facet_markers(nullptr),
          endo_marker(0), use_boundary_marker(false) {}
    
    void eval(Array<double>& values, const Array<double>& x) const override
    {
        // 没有单元信息时无法判定 marker，保守处理为 0（避免误刺激全域）
        values[0] = 0.0;
    }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& cell) const override
    {
        if (current_time < t_start || current_time > t_end) {
            values[0] = 0.0;
            return;
        }

        if (use_boundary_marker) {
            const bool is_near_endo = (endocardial_cells.find(cell.index) != endocardial_cells.end());
            values[0] = is_near_endo ? amp : 0.0;
        } else {
            // 兼容旧行为：无 marker 时全域刺激
            values[0] = amp;
        }
    }
    
    void update_time(double t_ms) { current_time = t_ms; }

private:
    void build_cell_to_boundary_map()
    {
        endocardial_cells.clear();
        if (!mesh || !facet_markers) {
            use_boundary_marker = false;
            return;
        }

        std::size_t num_endo_facets = 0;
        std::size_t num_endo_cells = 0;
        for (CellIterator cell(*mesh); !cell.end(); ++cell) {
            bool near_endo = false;
            for (FacetIterator facet(*cell); !facet.end(); ++facet) {
                if (!facet->exterior()) continue;
                if ((*facet_markers)[facet->index()] == endo_marker) {
                    near_endo = true;
                    ++num_endo_facets;
                    break;
                }
            }
            if (near_endo) {
                endocardial_cells.insert(cell->index());
                ++num_endo_cells;
            }
        }

        if (endocardial_cells.empty()) {
            LOG_F(WARNING,
                  "StimulusCurrentEP: no cells found near endocardial marker=%zu. "
                  "Fallback to global stimulus.", endo_marker);
            use_boundary_marker = false;
        } else {
            LOG_F(INFO,
                  "StimulusCurrentEP: endocardial marker=%zu, near-endo cells=%zu, matched exterior facets=%zu.",
                  endo_marker, num_endo_cells, num_endo_facets);
        }
    }

    double amp;
    double t_start;
    double t_end;
    double current_time;
    std::shared_ptr<const Mesh> mesh;
    std::shared_ptr<const MeshFunction<std::size_t>> facet_markers;
    std::size_t endo_marker;
    bool use_boundary_marker;
    std::unordered_set<std::size_t> endocardial_cells;
};


/**
 * 电生理求解器类
 * 
 * 实现单域方程的时间步进求解：
 * - 使用操作分裂：先求解 ODE（细胞模型），再求解 PDE（扩散）
 * - 支持多尺度时间步进（ODE 子循环）
 * - 与固体求解器接口：输出 Ca_i 和主动张力
 */
class ElectrophysiologySolver 
{
public:
    /**
     * 构造函数
     * 
     * @param mesh        FEniCS 网格
     * @param boundaries  边界标记（可选）
     * @param dt_pde_ms   PDE 时间步长 (ms)
     * @param dt_ode_ms   ODE 时间步长 (ms)
     */
    ElectrophysiologySolver(
        std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MeshFunction<std::size_t>> boundaries,
        double dt_pde_ms,
        double dt_ode_ms)
        : _mesh(mesh), _boundaries(boundaries),
          dt_pde_milliseconds(dt_pde_ms), dt_ode_milliseconds(dt_ode_ms),
          _endo_marker(2)
    {
        LOG_F(INFO, "初始化电生理求解器 (Monodomain + GPB + Land)");

        V_scalar = std::make_shared<Monodomain::Form_a_ep_FunctionSpace_0>(mesh);
        
        // 创建向量函数空间 (用于纤维方向插值, P1 向量空间)
        // 使用 FFC 生成的 CoefficientSpace_f0 (对应 VectorElement("Lagrange", tet, 1))
        V_vector = std::make_shared<Monodomain::CoefficientSpace_f0>(mesh);
        
        LOG_F(INFO, "  标量函数空间自由度: %zu", V_scalar->dim());
        
        // 初始化电生理变量
        Vm = std::make_shared<Function>(V_scalar);
        Vm_old = std::make_shared<Function>(V_scalar);
        I_ion = std::make_shared<Function>(V_scalar);
        I_stim = std::make_shared<Function>(V_scalar);
        
        // 初始化静息电位 (-81.5 mV = -0.0815 V)
        Constant V_rest(-0.0815455936324844);
        Vm->interpolate(V_rest);
        Vm_old->interpolate(V_rest);

        // Land版组织管理器接口: (function_space, dt_pde_ms, dt_ode_ms)
        tissue_manager = std::make_shared<GPBTissueManager>(
            V_scalar, dt_pde_ms, dt_ode_ms);

        dt_constant = std::make_shared<Constant>(dt_pde_ms / 1000.0);

        fiber_func = std::make_shared<Function>(V_vector);
        sheet_func = std::make_shared<Function>(V_vector);

        if (_boundaries) {
            stim_expr = std::make_shared<StimulusCurrentEP>(
                12.0, 500, 510, _mesh, _boundaries, _endo_marker); // 默认：内膜(marker=1)刺激
        } else {
            stim_expr = std::make_shared<StimulusCurrentEP>(12.0, 500, 510); // 无边界标记时回退全域刺激
        }
        I_stim->interpolate(*stim_expr);

        _initialized = false;
        LOG_F(INFO, "电生理求解器初始化完成（需要调用 setup_forms 完成配置）");
    }

    void set_fiber_directions(
        const std::string& fiber_0_xml,
        const std::string& fiber_1_xml,
        const std::string& fiber_2_xml,
        const std::string& sheet_0_xml,
        const std::string& sheet_1_xml,
        const std::string& sheet_2_xml)
    {
        LOG_F(INFO, "加载纤维方向...");
        
        auto f0 = std::make_shared<MeshFunction<double>>(_mesh, fiber_0_xml);
        auto f1 = std::make_shared<MeshFunction<double>>(_mesh, fiber_1_xml);
        auto f2 = std::make_shared<MeshFunction<double>>(_mesh, fiber_2_xml);
        auto s0 = std::make_shared<MeshFunction<double>>(_mesh, sheet_0_xml);
        auto s1 = std::make_shared<MeshFunction<double>>(_mesh, sheet_1_xml);
        auto s2 = std::make_shared<MeshFunction<double>>(_mesh, sheet_2_xml);
        
        fiber_expr = std::make_shared<FiberDirectionsEP>(f0, f1, f2);
        sheet_expr = std::make_shared<FiberDirectionsEP>(s0, s1, s2);
        
        fiber_func->interpolate(*fiber_expr);
        sheet_func->interpolate(*sheet_expr);
        
        LOG_F(INFO, "  纤维方向加载完成");
    }
    
    /**
     * 设置纤维方向（从数据目录自动拼接路径）
     * 
     * 期望目录中存在:
     *   fibers_0.xml, fibers_1.xml, fibers_2.xml
     *   sheets_0.xml, sheets_1.xml, sheets_2.xml
     * 
     * @param data_dir 数据目录路径，"/mnt/large2/gjh/realistic_left_ventricle"
     */
    void set_fiber_directions_from_dir(std::string& data_dir)
    {
        // 确保路径末尾有分隔符
        std::string dir = data_dir;
        if (!dir.empty() && dir.back() != '/') dir += "/";

        set_fiber_directions(
            dir + "fibers_0.xml",
            dir + "fibers_1.xml",
            dir + "fibers_2.xml",
            dir + "sheets_0.xml",
            dir + "sheets_1.xml",
            dir + "sheets_2.xml"
        );
    }

    /**
     * 设置纤维方向（直接传入 MeshFunction）
     */
    void set_fiber_directions(
        std::shared_ptr<FiberDirectionsEP> f0_expr,
        std::shared_ptr<FiberDirectionsEP> s0_expr)
    {
        fiber_expr = f0_expr;
        sheet_expr = s0_expr;
        fiber_func->interpolate(*fiber_expr);
        sheet_func->interpolate(*sheet_expr);
    }
    
    /**
     * 设置刺激电流参数
     */
    void set_stimulus(double amplitude, double t_start_ms, double t_end_ms)
    {
        if (_boundaries) {
            stim_expr = std::make_shared<StimulusCurrentEP>(
                amplitude, t_start_ms, t_end_ms, _mesh, _boundaries, _endo_marker);
            LOG_F(INFO, "设置内膜刺激(marker=%zu): %.2f μA/cm², [%.2f, %.2f] ms",
                  _endo_marker, amplitude, t_start_ms, t_end_ms);
        } else {
            stim_expr = std::make_shared<StimulusCurrentEP>(amplitude, t_start_ms, t_end_ms);
            LOG_F(INFO, "设置全域刺激(无边界标记): %.2f μA/cm², [%.2f, %.2f] ms",
                  amplitude, t_start_ms, t_end_ms);
        }
    }

    /**
     * 设置刺激电流参数（指定边界 marker，仅作用于该边界邻近单元）
     */
    void set_stimulus_on_marker(double amplitude, double t_start_ms, double t_end_ms,
                                std::size_t endo_marker)
    {
        _endo_marker = endo_marker;
        if (_boundaries) {
            stim_expr = std::make_shared<StimulusCurrentEP>(
                amplitude, t_start_ms, t_end_ms, _mesh, _boundaries, _endo_marker);
            LOG_F(INFO, "设置 marker 刺激(marker=%zu): %.2f μA/cm², [%.2f, %.2f] ms",
                  _endo_marker, amplitude, t_start_ms, t_end_ms);
        } else {
            stim_expr = std::make_shared<StimulusCurrentEP>(amplitude, t_start_ms, t_end_ms);
            LOG_F(WARNING, "set_stimulus_on_marker: boundary markers unavailable, fallback to global stimulus.");
        }
    }
    
    /**
     * 设置变分形式并创建求解器
     * 必须在 set_fiber_directions 之后调用
     */
    void setup_forms()
    {
        LOG_F(INFO, "设置电生理变分形式...");
        
        // 创建双线性形式和线性形式
        a_ep = std::make_shared<Monodomain::Form_a_ep>(V_scalar, V_scalar);
        L_ep = std::make_shared<Monodomain::Form_L_ep>(V_scalar);
        
        // 设置系数
        a_ep->f0 = fiber_func;
        a_ep->s0 = sheet_func;
        a_ep->dt = dt_constant;
        
        L_ep->Vm_old = Vm_old;
        L_ep->I_ion = I_ion;
        L_ep->I_stim = I_stim;
        L_ep->dt = dt_constant;
        
        // 无边界条件（自然边界条件：零通量）
        std::vector<std::shared_ptr<const DirichletBC>> bcs_ep;
        
        // 创建线性变分问题和求解器
        ep_problem = std::make_shared<LinearVariationalProblem>(a_ep, L_ep, Vm, bcs_ep);
        ep_solver = std::make_shared<LinearVariationalSolver>(ep_problem);
        
        // 求解器参数
        ep_solver->parameters["linear_solver"] = "cg";
        ep_solver->parameters["preconditioner"] = "petsc_amg";
        
        _initialized = true;
        LOG_F(INFO, "  电生理变分形式设置完成");
    }
    
    /**
     * 求解一个时间步
     * 
     * @param time_ms 当前时间 (ms)
     */
    void solve_timestep(double time_ms)
    {
        if (!_initialized) {
            throw std::runtime_error("ElectrophysiologySolver not initialized! Call setup_forms() first.");
        }
        
        // Step 1: 更新刺激电流
        stim_expr->update_time(time_ms);
        I_stim->interpolate(*stim_expr);

        // ODE子循环: GPB + Land ODE (XS/XW)
        tissue_manager->compute_ionic_current_subcycling(Vm_old, I_ion, time_ms);
        
        // Step 3: 求解 PDE
        ep_solver->solve();
        
        // Step 4: 更新 Vm_old
        *Vm_old = *Vm;
    }

    // ====== Solid 侧耦合接口 ======
    std::vector<double> get_Ca_i_field() const { return tissue_manager->get_Ca_i_vector(); }
    std::vector<double> get_XS_field() const { return tissue_manager->get_XS_vector(); }
    std::vector<double> get_XW_field() const { return tissue_manager->get_XW_vector(); }

    void update_mechanics_feedback(const std::vector<double>& lmbda,
                                   const std::vector<double>& zetas,
                                   const std::vector<double>& zetaw)
    {
        tissue_manager->update_mechanics_feedback(lmbda, zetas, zetaw);
    }

    // ====== 常规访问接口 ======
    std::shared_ptr<Function> get_Vm() const { return Vm; }
    std::shared_ptr<Function> get_Vm_old() const { return Vm_old; }
    std::shared_ptr<Function> get_I_ion() const { return I_ion; }
    std::shared_ptr<Function> get_I_stim() const { return I_stim; }
    std::shared_ptr<GPBTissueManager> get_tissue_manager() const { return tissue_manager; }
    std::shared_ptr<FunctionSpace> get_function_space() const { return V_scalar; }

    GPBTissueManager::Statistics get_statistics() const
    {
        return tissue_manager->get_statistics(Vm, I_ion);
    }

    GPBTissueManager::LandStats get_land_statistics() const
    {
        return tissue_manager->get_land_statistics();
    }

    GPBTissueManager::TensionStats get_tension_statistics() const
    {
        return tissue_manager->get_tension_statistics();
    }

    bool check_stability(double acceptable_Vm_range_mV = 200.0) const
    {
        return tissue_manager->check_stability(Vm, I_ion, acceptable_Vm_range_mV);
    }

    void set_Ca_i_to_function(std::shared_ptr<Function> Ca_i_func)
    {
        auto Ca_i_values = tissue_manager->get_Ca_i_vector();
        Ca_i_func->vector()->set_local(Ca_i_values);
        Ca_i_func->vector()->apply("insert");
    }

    void set_tension_to_function(std::shared_ptr<Function> T_func)
    {
        auto XS_values = tissue_manager->get_XS_vector();
        T_func->vector()->set_local(XS_values);
        T_func->vector()->apply("insert");
    }

private:
    std::shared_ptr<Mesh> _mesh;
    std::shared_ptr<MeshFunction<std::size_t>> _boundaries;
    std::size_t _endo_marker;
    std::shared_ptr<FunctionSpace> V_scalar;
    std::shared_ptr<FunctionSpace> V_vector;

    std::shared_ptr<Function> Vm;
    std::shared_ptr<Function> Vm_old;
    std::shared_ptr<Function> I_ion;
    std::shared_ptr<Function> I_stim;

    std::shared_ptr<Function> fiber_func;
    std::shared_ptr<Function> sheet_func;
    std::shared_ptr<FiberDirectionsEP> fiber_expr;
    std::shared_ptr<FiberDirectionsEP> sheet_expr;

    std::shared_ptr<StimulusCurrentEP> stim_expr;

    std::shared_ptr<Monodomain::Form_a_ep> a_ep;
    std::shared_ptr<Monodomain::Form_L_ep> L_ep;
    std::shared_ptr<LinearVariationalProblem> ep_problem;
    std::shared_ptr<LinearVariationalSolver> ep_solver;

    double dt_pde_milliseconds;
    double dt_ode_milliseconds;
    std::shared_ptr<Constant> dt_constant;

    std::shared_ptr<GPBTissueManager> tissue_manager;

    bool _initialized;
};

} // namespace dolfin

#endif // __ELECTROPHYSIOLOGY_SOLVER_H__
