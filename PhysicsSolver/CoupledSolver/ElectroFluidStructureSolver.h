/**
 * @file ElectroFluidStructureSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 电-流-固三场耦合求解器
 * @version 0.2
 * @date 2026-03-11
 * 
 * @copyright Copyright (c) 2026 Ma Pengfei
 * 
 * 将电生理（Electrophysiology）、固体力学（Solid）、流体力学（Fluid）
 * 三个物理场耦合在一起，实现心脏电-流-固耦合模拟。
 * 
 * 数据流:
 *   EP(节点级) --Ca_i--> node_to_cell --> 固体(单元级) --calculate_T--> T
 *   固体(单元级) --lambda--> cell_to_node --> EP(节点级) --NHS--> 主动收缩
 *   固体 <--IBM--> 流体 (浸没边界法)
 */

#ifndef __ELECTRO_FLUID_STRUCTURE_SOLVER_H__
#define __ELECTRO_FLUID_STRUCTURE_SOLVER_H__

#include <dolfin.h>
#include <loguru/loguru.hpp>

#include "../ElectrophysiologySolver/ElectrophysiologySolver.h"
#include "../SolidSolver/ActiveLeftVentricle/ActiveLeftVentricleSolver.h"
#include "../SolidSolver/ActiveLeftVentricle/ActiveContraction.h"
#include "../FluidSolver/FluidSolver.h"
#include "../ImmersedBoundaryMethod/ImmersedBoundaryMethod2.h"
#include "../ImmersedBoundaryMethod/ElerianLagrangianInteraction2.h"

#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/StdVector.h>

namespace dolfin {


template<typename SolidSolverType, typename FluidSolverType, typename VectorType>
class ElectroFluidStructureSolver 
{
public:
    // ═══════════════════════════════════════════════════════════════════
    // 插值工具：节点 ↔ 单元 数据转换
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 节点级数据 → 单元级数据（取单元所有顶点的平均值）
     * 
     * 适用于：Ca_i (EP 节点级) → 固体求解器 (单元级 MeshFunction)
     * 
     * @param mesh          dolfin 有限元网格
     * @param node_values   节点级数据 (大小 = num_vertices)
     * @return              单元级数据 (大小 = num_cells)
     */
    static std::vector<double> node_to_cell_average(
        const Mesh& mesh, 
        const std::vector<double>& node_values)
    {
        const std::size_t num_cells = mesh.num_cells();
        std::vector<double> cell_values(num_cells, 0.0);
        
        for (CellIterator cell(mesh); !cell.end(); ++cell)
        {
            double sum = 0.0;
            int count = 0;
            for (VertexIterator v(*cell); !v.end(); ++v)
            {
                const std::size_t vi = v->index();
                if (vi < node_values.size())
                {
                    sum += node_values[vi];
                    ++count;
                }
            }
            cell_values[cell->index()] = (count > 0) ? sum / count : 0.0;
        }
        return cell_values;
    }
    
    /**
     * 单元级数据 → 节点级数据（取共享该节点的所有单元的平均值）
     * 
     * 适用于：lambda_f (固体单元级) → EP 节点级
     * 
     * @param mesh          dolfin 有限元网格
     * @param cell_values   单元级数据 (大小 = num_cells)
     * @return              节点级数据 (大小 = num_vertices)
     */
    static std::vector<double> cell_to_node_average(
        const Mesh& mesh, 
        const std::vector<double>& cell_values)
    {
        const std::size_t num_vertices = mesh.num_vertices();
        std::vector<double> node_values(num_vertices, 0.0);
        std::vector<int> node_counts(num_vertices, 0);
        
        for (CellIterator cell(mesh); !cell.end(); ++cell)
        {
            const double cv = cell_values[cell->index()];
            for (VertexIterator v(*cell); !v.end(); ++v)
            {
                const std::size_t vi = v->index();
                node_values[vi] += cv;
                node_counts[vi]++;
            }
        }
        
        for (std::size_t i = 0; i < num_vertices; ++i)
        {
            if (node_counts[i] > 0)
                node_values[i] /= node_counts[i];
        }
        return node_values;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 构造与初始化
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 构造函数
     * 
     * @param solid_mesh    固体网格 (ImmersedMesh, Lagrangian)
     * @param fluid_mesh    流体网格 (BackgroundMesh2, Eulerian)
     * @param solid_solver  固体求解器 (ActiveLeftVentricleSolver)
     * @param fluid_solver  流体求解器 (ProjectionSchemeGPU)
     * @param ep_solver     电生理求解器 (ElectrophysiologySolver)
     */
    ElectroFluidStructureSolver(
        std::shared_ptr<ImmersedMesh> solid_mesh,
        std::shared_ptr<BackgroundMesh2> fluid_mesh,
        std::shared_ptr<SolidSolverType> solid_solver,
        std::shared_ptr<FluidSolverType> fluid_solver,
        std::shared_ptr<ElectrophysiologySolver> ep_solver)
        : _solid_mesh(solid_mesh),
          _fluid_mesh(fluid_mesh),
          _solid_solver(solid_solver),
          _fluid_solver(fluid_solver),
          _ep_solver(ep_solver),
          _ibm_solver(std::make_shared<ImmersedBoundaryMethod2<SolidSolverType, FluidSolverType, VectorType>>(
              solid_mesh, fluid_mesh, solid_solver, fluid_solver)),
          _t(0.0), _dt(0.0),
          _t_end_diastole(0.5),     // 舒张期结束时间 [s]
          _ep_enabled(true)
    {
        // 获取 dolfin mesh 用于插值
        _dolfin_mesh = _solid_mesh->get_dolfin_mesh();
        
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "初始化电-流-固三场耦合求解器");
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "  固体网格节点数: %zu", _dolfin_mesh->num_vertices());
        LOG_F(INFO, "  固体网格单元数: %zu", _dolfin_mesh->num_cells());
        LOG_F(INFO, "  电生理自由度:   %zu", ep_solver->get_function_space()->dim());
        LOG_F(INFO, "  舒张期结束:     %.3f s", _t_end_diastole);
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 初始化 Newton 求解器（用于 IBM 隐式格式）
        auto bicgstab = std::make_shared<BiCGSTAB<VectorType>>(_ibm_solver->unkown_size() / 3);
        bicgstab->set_tolerance(1e-3);
        bicgstab->set_max_iteration(20);
        _newton_solver = std::make_shared<::NewtonSolver<VectorType>>(bicgstab);
        _newton_solver->max_nonlinear_tolerance = 1e-5;
        _newton_solver->max_nonlinear_iteration = 100;
        
        // 初始化位置向量
        _xn = std::make_shared<VectorType>();
        _xn_1 = std::make_shared<VectorType>();
        _xn_new = std::make_shared<VectorType>();
        _xn->resize(_ibm_solver->unkown_size() / 3);
        _xn_1->resize(_ibm_solver->unkown_size() / 3);
        _xn_new->resize(_ibm_solver->unkown_size() / 3);
        
        _ibm_solver->get_solid_positions(*_xn);
        *_xn_1 = *_xn;
        *_xn_new = *_xn;
        _dt_ratio = 1.0;
        
        // 初始化 ActiveContraction 查表数据（模式 B 需要）
        ActiveContraction::read_GPB_data();
        
        LOG_F(INFO, "电-流-固三场耦合求解器初始化完成");
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 参数设置
    // ═══════════════════════════════════════════════════════════════════
    
    void set_dt(double dt) { _dt = dt; }
    double get_dt() const { return _dt; }
    double get_t() const { return _t; }
    
    void set_t_end_diastole(double t) { _t_end_diastole = t; }
    double get_t_end_diastole() const { return _t_end_diastole; }
    
    /** 启用/禁用电生理耦合（禁用时退化为原始 FSI + 查表 Ca_i） */
    void set_ep_enabled(bool enabled) { _ep_enabled = enabled; }
    bool is_ep_enabled() const { return _ep_enabled; }

    // ═══════════════════════════════════════════════════════════════════
    // 核心时间步进
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 执行一个完整的电-流-固耦合时间步
     * 
     * 时序:
     *   1. 电生理 PDE/ODE 求解 (多个子步)
     *   2. Ca_i → node_to_cell → 固体 (电-力耦合)
     *   3. calculate_T (主动收缩张力)
     *   4. lambda → cell_to_node → EP (力-电反馈)
     *   5. IBM Newton solve (流固耦合)
     *   6. advance (更新位置/速度)
     * 
     * @param t   当前时间 [s]
     * @param dt  力学时间步长 [s]
     */
    void solve_timestep(double t, double dt)
    {
        LOG_SCOPE_FUNCTION(INFO);
        LOG_F(INFO, "电-流-固耦合步: t = %.6e s, dt = %.6e s", t, dt);
        
        _t = t;
        _dt = dt;
        
        // ═══════════════════════════════════════════════════════════
        // Step 1: 电生理求解（多个 PDE 步）
        // ═══════════════════════════════════════════════════════════
        if (_ep_enabled && t >= _t_end_diastole)
        {
            double t_ms = t * 1000.0;
            double dt_ep_ms = _ep_solver->get_tissue_manager()->get_dt_pde_ms();
            
            int n_ep_steps = static_cast<int>(std::round(dt * 1000.0 / dt_ep_ms));
            if (n_ep_steps < 1) n_ep_steps = 1;
            
            LOG_F(INFO, "电生理子步: %d 步 × %.4f ms", n_ep_steps, dt_ep_ms);
            
            for (int ep_step = 0; ep_step < n_ep_steps; ++ep_step) {
                double t_ep_ms_current = t_ms + ep_step * dt_ep_ms;
                _ep_solver->solve_timestep(t_ep_ms_current);
            }
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 2: 主动收缩张力计算
        // ═══════════════════════════════════════════════════════════
        ActiveContraction::t_end_diastole = _t_end_diastole;
        ActiveContraction::time_current = t;
        
        if (t >= _t_end_diastole)
        {
            if (_ep_enabled)
            {
                // 模式 A: 电-力耦合 — Ca_i 来自电生理
                // EP (节点级 Ca_i) → node_to_cell → 固体 (单元级 MeshFunction)
                auto Ca_i_node = _ep_solver->get_Ca_i_field();    // 节点级 [μM]
                auto Ca_i_cell = node_to_cell_average(
                    *_dolfin_mesh, Ca_i_node);                     // 单元级
                _solid_solver->set_Ca_i_from_electrophysiology(Ca_i_cell);
            }
            else
            {
                // 模式 B: 原始模式 — Ca_i 来自 dat 文件查表
                ActiveContraction::cai_current_calculation(t, dt);
            }
            
            // 计算主动收缩张力 T（两种模式都需要）
            _solid_solver->contraction->calculate_T(t, dt);
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 3: 力学状态反馈到电生理
        // ═══════════════════════════════════════════════════════════
        if (_ep_enabled)
        {
            auto [lambda_cell, dlambda_dt_cell] = _solid_solver->get_mechanical_state();
            
            if (!lambda_cell.empty()) {
                auto lambda_node = cell_to_node_average(
                    *_dolfin_mesh, lambda_cell);
                auto dlambda_dt_node = cell_to_node_average(
                    *_dolfin_mesh, dlambda_dt_cell);
                _ep_solver->set_mechanical_state(lambda_node, dlambda_dt_node);
            }
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 4: 流固耦合求解（浸没边界法 + Newton 迭代）
        // ═══════════════════════════════════════════════════════════
        
        // 预测初值: x_new = (1+1/dt_ratio)*xn - (1/dt_ratio)*xn_1
        _xn_new->axpy(-(1.0 + _dt_ratio) / _dt_ratio, *_xn, *_xn_1);
        _xn_new->axpy(-1.0 - _dt_ratio, *_xn_new, *_xn_new);
        
        _ibm_solver->set_dt(dt * _dt_ratio);
        auto nonlinear_result = _newton_solver->Solve(_ibm_solver, _xn_new, nullptr);
        
        // 更新位置
        *_xn_1 = *_xn;
        *_xn = *_xn_new;
        _ibm_solver->advance(*_xn_new);
        
        // 更新时间
        _t += dt;
        
        LOG_F(INFO, "时间步完成: t_new = %.6e s, Newton 迭代 = %d, 残差 = %.2e",
              _t, nonlinear_result.second.second, nonlinear_result.second.first);
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 便捷方法：仅求解部分物理场
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 仅求解电生理（不进行流固耦合）
     */
    void solve_electrophysiology_only(double t_ms)
    {
        _ep_solver->solve_timestep(t_ms);
    }
    
    /**
     * 仅求解固体力学（使用电生理结果），不进行 IBM
     */
    void solve_solid_with_ep(double t, double dt)
    {
        auto Ca_i_node = _ep_solver->get_Ca_i_field();
        auto Ca_i_cell = node_to_cell_average(*_dolfin_mesh, Ca_i_node);
        _solid_solver->set_Ca_i_from_electrophysiology(Ca_i_cell);
        _solid_solver->contraction->calculate_T(t, dt);
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 访问器
    // ═══════════════════════════════════════════════════════════════════
    
    std::shared_ptr<ElectrophysiologySolver> get_ep_solver() const { return _ep_solver; }
    std::shared_ptr<SolidSolverType> get_solid_solver() const { return _solid_solver; }
    std::shared_ptr<FluidSolverType> get_fluid_solver() const { return _fluid_solver; }
    
    std::shared_ptr<ImmersedBoundaryMethod2<SolidSolverType, FluidSolverType, VectorType>> 
    get_ibm_solver() const { return _ibm_solver; }
    
    std::shared_ptr<Mesh> get_dolfin_mesh() const { return _dolfin_mesh; }
    
    // ═══════════════════════════════════════════════════════════════════
    // 统计与诊断
    // ═══════════════════════════════════════════════════════════════════
    
    GPBTissueManager::Statistics get_ep_statistics() const
    {
        return _ep_solver->get_statistics();
    }
    
    GPBTissueManager::TensionStats get_tension_statistics() const
    {
        return _ep_solver->get_tension_statistics();
    }
    
    bool check_stability() const
    {
        return _ep_solver->check_stability();
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 文件输出
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 输出当前状态到文件
     */
    void output_state(
        File& displacement_file,
        File& Vm_file,
        File& tension_file)
    {
        // 输出位移
        displacement_file << *(_solid_solver->X_current);
        
        // 输出膜电位
        Vm_file << *(_ep_solver->get_Vm());
        
        // 输出张力（创建临时函数）
        auto T_func = std::make_shared<Function>(_ep_solver->get_function_space());
        _ep_solver->set_tension_to_function(T_func);
        tension_file << *T_func;
    }
    
    /**
     * 输出边界点位置（用于压力-容积曲线）
     */
    void record_boundary_points()
    {
        _solid_solver->record_boundary_points();
    }

private:
    // 网格
    std::shared_ptr<ImmersedMesh> _solid_mesh;
    std::shared_ptr<BackgroundMesh2> _fluid_mesh;
    std::shared_ptr<Mesh> _dolfin_mesh;   // = _solid_mesh->get_dolfin_mesh()
    
    // 求解器
    std::shared_ptr<SolidSolverType> _solid_solver;
    std::shared_ptr<FluidSolverType> _fluid_solver;
    std::shared_ptr<ElectrophysiologySolver> _ep_solver;
    std::shared_ptr<ImmersedBoundaryMethod2<SolidSolverType, FluidSolverType, VectorType>> _ibm_solver;
    
    // Newton 求解器（用于 IBM 隐式求解）
    std::shared_ptr<::NewtonSolver<VectorType>> _newton_solver;
    
    // 位置向量（用于 Newton 迭代）
    std::shared_ptr<VectorType> _xn;       // x_n (当前步)
    std::shared_ptr<VectorType> _xn_1;     // x_{n-1} (上一步)
    std::shared_ptr<VectorType> _xn_new;   // x_{n+1} (Newton 迭代结果)
    double _dt_ratio;                       // 自适应步长比
    
    // 时间参数
    double _t;
    double _dt;
    double _t_end_diastole;   // 舒张期结束时间 [s]
    
    // 电生理开关
    bool _ep_enabled;
};

} // namespace dolfin

#endif // __ELECTRO_FLUID_STRUCTURE_SOLVER_H__