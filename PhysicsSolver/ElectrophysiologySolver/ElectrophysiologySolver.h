#ifndef __ELECTROPHYSIOLOGY_SOLVER_H__
#define __ELECTROPHYSIOLOGY_SOLVER_H__

#include <dolfin.h>
#include <loguru/loguru.hpp>
#include "GPB_cell_model.h"
#include "GPBTissueManager_Land.h"
#include "Monodomain.h"  // FFC 生成

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

/**
 * 刺激电流 Expression
 */
class StimulusCurrentEP : public Expression
{
public:
    StimulusCurrentEP(double amplitude, double t_start_ms, double t_end_ms)
        : Expression(), amp(amplitude), t_start(t_start_ms), t_end(t_end_ms), 
          current_time(0.0) {}
    
    void eval(Array<double>& values, const Array<double>& x) const override
    {
        if (current_time >= t_start && current_time <= t_end) {
            values[0] = amp;
        } else {
            values[0] = 0.0;
        }
    }
    
    void update_time(double t_ms) { current_time = t_ms; }

private:
    double amp;
    double t_start;
    double t_end;
    double current_time;
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
          dt_pde_milliseconds(dt_pde_ms), dt_ode_milliseconds(dt_ode_ms)
    {
        LOG_F(INFO, "初始化电生理求解器 (Monodomain + GPB)");
        
        // 创建标量函数空间 (用于膜电位)
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
        LOG_F(INFO, "  初始静息电位: %.2f mV", Vm->vector()->max() * 1000.0);
        
        // 初始化 GPB 组织管理器
        tissue_manager = std::make_shared<GPBTissueManager>(
            V_scalar, mesh, dt_pde_ms, dt_ode_ms
        );
        
        // 时间步长常量
        dt_constant = std::make_shared<Constant>(dt_pde_ms / 1000.0);  // 转换为秒
        
        // 纤维方向函数（需要外部设置）
        fiber_func = std::make_shared<Function>(V_vector);
        sheet_func = std::make_shared<Function>(V_vector);
        
        // 默认刺激（无刺激）
        stim_expr = std::make_shared<StimulusCurrentEP>(0.0, 0.0, 0.0);
        I_stim->interpolate(*stim_expr);
        
        _initialized = false;
        LOG_F(INFO, "电生理求解器初始化完成（需要调用 setup_forms 完成配置）");
    }
    
    /**
     * 设置纤维方向（从 XML 文件加载）
     */
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
        stim_expr = std::make_shared<StimulusCurrentEP>(amplitude, t_start_ms, t_end_ms);
        LOG_F(INFO, "设置刺激电流: %.2f μA/cm², [%.2f, %.2f] ms", 
              amplitude, t_start_ms, t_end_ms);
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
        
        // Step 2: 计算离子电流（ODE 子循环）
        tissue_manager->compute_ionic_current_subcycling(Vm_old, I_ion, time_ms);
        
        // Step 3: 求解 PDE
        ep_solver->solve();
        
        // Step 4: 更新 Vm_old
        *Vm_old = *Vm;
    }
    
    /**
     * 获取钙离子浓度场 (用于力学求解器)
     * @return 节点级钙离子浓度向量 (μM)
     */
    std::vector<double> get_Ca_i_field() const
    {
        return tissue_manager->get_Ca_i_vector();
    }
    
    /**
     * 获取主动张力场
     * @return 节点级主动张力向量 (kPa)
     */
    std::vector<double> get_active_tension_field() const
    {
        return tissue_manager->get_active_tension();
    }
    
    /**
     * 获取膜电位函数 (用于输出)
     */
    std::shared_ptr<Function> get_Vm() const { return Vm; }
    
    /**
     * 获取膜电位旧值函数
     */
    std::shared_ptr<Function> get_Vm_old() const { return Vm_old; }
    
    /**
     * 获取离子电流函数
     */
    std::shared_ptr<Function> get_I_ion() const { return I_ion; }
    
    /**
     * 获取刺激电流函数
     */
    std::shared_ptr<Function> get_I_stim() const { return I_stim; }
    
    /**
     * 获取 GPB 组织管理器（用于高级操作）
     */
    std::shared_ptr<GPBTissueManager> get_tissue_manager() const { return tissue_manager; }
    
    /**
     * 获取标量函数空间
     */
    std::shared_ptr<FunctionSpace> get_function_space() const { return V_scalar; }
    
    /**
     * 设置力学状态（从固体求解器接收）
     */
    void set_mechanical_state(const std::vector<double>& lambda,
                              const std::vector<double>& dlambda_dt)
    {
        tissue_manager->set_mechanical_state(lambda, dlambda_dt);
    }
    
    /**
     * 获取统计信息
     */
    GPBTissueManager::Statistics get_statistics() const
    {
        return tissue_manager->get_statistics(Vm, I_ion);
    }
    
    /**
     * 获取张力统计信息
     */
    GPBTissueManager::TensionStats get_tension_statistics() const
    {
        return tissue_manager->get_tension_statistics();
    }
    
    /**
     * 检查数值稳定性
     */
    bool check_stability(double acceptable_Vm_range_mV = 200.0) const
    {
        return tissue_manager->check_stability(Vm, I_ion, acceptable_Vm_range_mV);
    }
    
    /**
     * 将主动张力设置到 FEniCS Function（用于可视化）
     */
    void set_tension_to_function(std::shared_ptr<Function> T_func)
    {
        tissue_manager->set_tension_function(T_func);
    }
    
    /**
     * 将钙离子浓度设置到 FEniCS Function（用于可视化）
     */
    void set_Ca_i_to_function(std::shared_ptr<Function> Ca_i_func)
    {
        auto Ca_i_values = tissue_manager->get_Ca_i_vector();
        Ca_i_func->vector()->set_local(Ca_i_values);
        Ca_i_func->vector()->apply("insert");
    }
    
    /**
     * 插值主动张力到单元级（用于固体求解器 UFL 形式）
     */
    void interpolate_tension_to_cells()
    {
        tissue_manager->interpolate_tension_to_cells();
    }
    
    /**
     * 获取单元级张力 MeshFunction
     */
    std::shared_ptr<MeshFunction<double>> get_T_cell_meshfunction() const
    {
        return tissue_manager->get_T_cell_meshfunction();
    }

private:
    // FEniCS 对象
    std::shared_ptr<Mesh> _mesh;
    std::shared_ptr<MeshFunction<std::size_t>> _boundaries;
    std::shared_ptr<FunctionSpace> V_scalar;
    std::shared_ptr<FunctionSpace> V_vector;
    
    // 电生理变量
    std::shared_ptr<Function> Vm;
    std::shared_ptr<Function> Vm_old;
    std::shared_ptr<Function> I_ion;
    std::shared_ptr<Function> I_stim;
    
    // 纤维方向
    std::shared_ptr<Function> fiber_func;
    std::shared_ptr<Function> sheet_func;
    std::shared_ptr<FiberDirectionsEP> fiber_expr;
    std::shared_ptr<FiberDirectionsEP> sheet_expr;
    
    // 刺激电流
    std::shared_ptr<StimulusCurrentEP> stim_expr;
    
    // 变分形式
    std::shared_ptr<Monodomain::Form_a_ep> a_ep;
    std::shared_ptr<Monodomain::Form_L_ep> L_ep;
    std::shared_ptr<LinearVariationalProblem> ep_problem;
    std::shared_ptr<LinearVariationalSolver> ep_solver;
    
    // 时间参数
    double dt_pde_milliseconds;
    double dt_ode_milliseconds;
    std::shared_ptr<Constant> dt_constant;
    
    // GPB 组织管理器
    std::shared_ptr<GPBTissueManager> tissue_manager;
    
    // 初始化标志
    bool _initialized;
};

} // namespace dolfin

#endif // __ELECTROPHYSIOLOGY_SOLVER_H__