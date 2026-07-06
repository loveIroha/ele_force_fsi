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
 */

#ifndef __ELECTRO_FLUID_STRUCTURE_SOLVER_H__
#define __ELECTRO_FLUID_STRUCTURE_SOLVER_H__

#include <dolfin.h>
#include <io/include/io/loguru.hpp>

#include "../ElectrophysiologySolver/ElectrophysiologySolver.h"
// #include "../SolidSolver/ActiveLeftVentricle/ActiveLeftVentricleSolver.h"
// #include "../SolidSolver/ActiveLeftVentricle/ActiveContraction.h"
#include "../FluidSolver/FluidSolver.h"
#include "../ImmersedBoundaryMethod/ImmersedBoundaryMethod3D.h"
#include "../ImmersedBoundaryMethod/ElerianLagrangianInteraction3D.h"

#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/StdVector.h>
#include <unordered_map>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace dolfin {

// 收缩期初始时间，同时也是舒张期结束时间 [s]。
static constexpr double kSystoleStartTime = 0.5;
static constexpr double t_period_con = 0.80;

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
    template <typename USolid, typename UFluid>
    ElectroFluidStructureSolver(
        std::shared_ptr<ImmersedMeshP1> solid_mesh,
        std::shared_ptr<BackgroundMesh3D<3>> fluid_mesh,
        std::shared_ptr<USolid> solid_solver,
        std::shared_ptr<UFluid> fluid_solver,
        std::shared_ptr<ElectrophysiologySolver> ep_solver)
        : _solid_mesh(solid_mesh),
          _fluid_mesh(fluid_mesh),
          _solid_solver(std::dynamic_pointer_cast<SolidSolverType>(solid_solver)),
          _fluid_solver(std::dynamic_pointer_cast<FluidSolverType>(fluid_solver)),
          _ep_solver(ep_solver),
          _ibm_solver(std::make_shared<ImmersedBoundaryMethod<FluidSolverType, SolidSolverType>>(
              fluid_mesh, std::dynamic_pointer_cast<FluidSolverType>(fluid_solver), solid_mesh, std::dynamic_pointer_cast<SolidSolverType>(solid_solver))),
          _t(0.0), _dt(0.0),
          _t_end_diastole(kSystoleStartTime),     // 舒张期结束时间 [s]
          _t_period(t_period_con),  
          _ep_enabled(true)
    {
        // 获取 dolfin mesh 用于插值
        _dolfin_mesh = _solid_mesh->get_dolfin_mesh();

        if (!_solid_solver) {
            throw std::runtime_error("ElectroFluidStructureSolver: solid solver cast failed");
        }
        if (!_fluid_solver) {
            throw std::runtime_error("ElectroFluidStructureSolver: fluid solver cast failed");
        }
        if (!_ep_solver) {
            throw std::runtime_error("ElectroFluidStructureSolver: ep solver is null");
        }
        if (!_dolfin_mesh) {
            throw std::runtime_error("ElectroFluidStructureSolver: solid dolfin mesh is null");
        }

        const std::size_t solid_scalar_dofs = _solid_mesh->num_dofs();
        const std::size_t solid_vector_dofs = _solid_solver->V ? _solid_solver->V->dim() : 0;
        if (solid_vector_dofs != solid_scalar_dofs * 3) {
            throw std::runtime_error(
                "ElectroFluidStructureSolver: solid dof mismatch, mesh scalar dofs=" +
                std::to_string(solid_scalar_dofs) + ", solver vector dofs=" +
                std::to_string(solid_vector_dofs));
        }

        auto make_key = [](double x, double y, double z) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%.9f_%.9f_%.9f", x, y, z);
            return std::string(buf);
        };

        auto build_coord_map = [&](const std::vector<double>& all_coords,
                                   const std::vector<int>& dofs,
                                   std::size_t all_dofs,
                                   const char* label) {
            if (all_coords.size() != all_dofs * 3) {
                throw std::runtime_error(
                    std::string("ElectroFluidStructureSolver: coord size mismatch for " ) + label +
                    ", coords=" + std::to_string(all_coords.size() / 3) +
                    ", dofs=" + std::to_string(all_dofs));
            }

            std::unordered_map<std::string, std::size_t> coord_map;
            coord_map.reserve(dofs.size());
            for (int dof : dofs) {
                if (dof < 0 || static_cast<std::size_t>(dof) >= all_dofs) {
                    throw std::runtime_error(
                        std::string("ElectroFluidStructureSolver: dof index out of range for " ) + label);
                }
                const std::size_t idx = static_cast<std::size_t>(dof);
                coord_map.emplace(
                    make_key(all_coords[3 * idx], all_coords[3 * idx + 1], all_coords[3 * idx + 2]),
                    idx);
            }
            return coord_map;
        };

        const std::size_t vector_dofs = _solid_solver->V->dim();
        std::vector<double> vector_coords = _solid_solver->V->tabulate_dof_coordinates();
        if (vector_coords.size() != vector_dofs * 3) {
            throw std::runtime_error(
                "ElectroFluidStructureSolver: vector coord mismatch, coords=" +
                std::to_string(vector_coords.size() / 3) + ", dofs=" +
                std::to_string(vector_dofs));
        }

        auto Vx = _solid_solver->V->sub(0);
        auto Vy = _solid_solver->V->sub(1);
        auto Vz = _solid_solver->V->sub(2);

        if (Vx->dim() != solid_scalar_dofs || Vy->dim() != solid_scalar_dofs || Vz->dim() != solid_scalar_dofs) {
            throw std::runtime_error(
                "ElectroFluidStructureSolver: subspace dof mismatch with scalar mesh dofs");
        }

        const std::vector<int> dofs_x = Vx->dofmap()->dofs();
        const std::vector<int> dofs_y = Vy->dofmap()->dofs();
        const std::vector<int> dofs_z = Vz->dofmap()->dofs();

        if (dofs_x.size() != solid_scalar_dofs || dofs_y.size() != solid_scalar_dofs || dofs_z.size() != solid_scalar_dofs) {
            throw std::runtime_error(
                "ElectroFluidStructureSolver: subspace dofmap size mismatch with scalar mesh dofs");
        }

        auto map_x = build_coord_map(vector_coords, dofs_x, vector_dofs, "Vx");
        auto map_y = build_coord_map(vector_coords, dofs_y, vector_dofs, "Vy");
        auto map_z = build_coord_map(vector_coords, dofs_z, vector_dofs, "Vz");

        const auto& imm_coords = _solid_mesh->get_dof_coordinates();
        if (imm_coords.size() != solid_scalar_dofs) {
            throw std::runtime_error(
                "ElectroFluidStructureSolver: ImmersedMesh dof mismatch, got " +
                std::to_string(imm_coords.size()) + ", expected " + std::to_string(solid_scalar_dofs));
        }

        std::vector<std::size_t> imm_to_dolfin_vector;
        imm_to_dolfin_vector.resize(solid_scalar_dofs * 3);

        for (std::size_t i = 0; i < imm_coords.size(); ++i) {
            const auto key = make_key(imm_coords[i].x, imm_coords[i].y, imm_coords[i].z);
            auto itx = map_x.find(key);
            auto ity = map_y.find(key);
            auto itz = map_z.find(key);
            if (itx == map_x.end() || ity == map_y.end() || itz == map_z.end()) {
                throw std::runtime_error(
                    "ElectroFluidStructureSolver: dof coordinate mismatch at index " + std::to_string(i));
            }
            imm_to_dolfin_vector[3 * i] = itx->second;
            imm_to_dolfin_vector[3 * i + 1] = ity->second;
            imm_to_dolfin_vector[3 * i + 2] = itz->second;
        }

        _solid_solver->set_dofmap_imm_to_dolfin_vector(imm_to_dolfin_vector);
        
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "初始化电-流-固三场耦合求解器");
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "  固体网格节点数: %zu", _dolfin_mesh->num_vertices());
        LOG_F(INFO, "  固体网格单元数: %zu", _dolfin_mesh->num_cells());
        LOG_F(INFO, "  电生理自由度:   %zu", ep_solver->get_function_space()->dim());
        LOG_F(INFO, "  舒张期结束:     %.3f s", _t_end_diastole);
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 初始化 Newton 求解器（用于 IBM 隐式格式）
        auto bicgstab = std::make_shared<BiCGSTAB<VectorType>>(1);
        bicgstab->set_tolerance(1e-3);
        bicgstab->set_max_iteration(20);
        _newton_solver = std::make_shared<::NewtonSolver<VectorType>>(bicgstab);
        _newton_solver->max_nonlinear_tolerance = 1e-5;
        _newton_solver->max_nonlinear_iteration = 100;
        
        // 初始化位置向量
        _xn = std::make_shared<VectorType>();
        _xn_1 = std::make_shared<VectorType>();
        _xn_new = std::make_shared<VectorType>();
        
        _xn->resize(_ibm_solver->_solid_displacement.size());_xn_1->resize(_xn->size());_xn_new->resize(_xn->size());for(size_t i=0;i<_xn->size();++i){_xn->data()[i]=_ibm_solver->_solid_displacement[i];_xn_1->data()[i]=_xn->data()[i];_xn_new->data()[i]=_xn->data()[i];}
        
        
        _dt_ratio = 1.0;
        
        
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

    void set_t_period(double t)
    {
        _t_period = t;
        if (_ep_solver) {
            _ep_solver->set_stimulus_period(_t_period * 1000.0);
        }
    }
    double get_t_period() const { return _t_period; }
    
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
        _solid_solver->_t = _t;
        _solid_solver->_dt = _dt;
        _solid_solver->_t = _t;
        _solid_solver->_dt = _dt;
        _solid_solver->_t = _t;
        _solid_solver->_dt = _dt;

        if (_ibm_solver->_solid_displacement.size() != _solid_mesh->num_dofs()) {
            throw std::runtime_error(
                "solve_timestep: solid displacement size mismatch, got " +
                std::to_string(_ibm_solver->_solid_displacement.size()) +
                ", expected " + std::to_string(_solid_mesh->num_dofs()));
        }

        const bool ep_active = is_ep_active(t);
        // ═══════════════════════════════════════════════════════════
        // Step 1: 电生理/Land 求解（多个 EP/Land 子步）
        // ═══════════════════════════════════════════════════════════
        if (_ep_enabled && ep_active)
        {
            double t_ms = t * 1000.0;
            double dt_ep_ms = _ep_solver->get_tissue_manager()->get_dt_pde_ms();
            
            int n_ep_steps = static_cast<int>(std::round(dt * 1000.0 / dt_ep_ms));
            if (n_ep_steps < 1) n_ep_steps = 1;
            
            LOG_F(INFO, "电生理+Land子步: %d 步 × %.4f ms", n_ep_steps, dt_ep_ms);
            
            for (int ep_step = 0; ep_step < n_ep_steps; ++ep_step) {
                double t_ep_ms_current = t_ms + ep_step * dt_ep_ms;
                _ep_solver->solve_timestep(t_ep_ms_current);
            }
        }
        else if (_ep_enabled)
        {
            LOG_F(INFO, "舒张期相位 local_t=%.6e s < t_end_diastole=%.6e s，跳过 EP/Land 演化。",
                  cycle_time(t), _t_end_diastole);
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 2: 主动收缩张力相关的横桥传递 (电 -> 力)
        // ═══════════════════════════════════════════════════════════
        if (_ep_enabled)
        {
            // Land 提供横桥状态 (XS, XW) -> 固体；刺激前显式置零，保证 Ta=0。
            auto XS_vec = ep_active ? _ep_solver->get_XS_field()
                                    : std::vector<double>(_solid_mesh->num_dofs(), 0.0);
            auto XW_vec = ep_active ? _ep_solver->get_XW_field()
                                    : std::vector<double>(_solid_mesh->num_dofs(), 0.0);
            if (XS_vec.size() != _solid_mesh->num_dofs() || XW_vec.size() != _solid_mesh->num_dofs()) {
                throw std::runtime_error(
                    "solve_timestep: XS/XW size mismatch, XS=" + std::to_string(XS_vec.size()) +
                    ", XW=" + std::to_string(XW_vec.size()) +
                    ", expected " + std::to_string(_solid_mesh->num_dofs()));
            }
            _solid_solver->set_land_crossbridge_states(XS_vec, XW_vec);
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 3: 流固耦合求解（与 ImmersedBoundaryMethod3D 同步：显式格式）
        // ═══════════════════════════════════════════════════════════

        // 与原始 IBM 显式流程一致：本步求解使用 t^{n+1}=t+dt 的时间标签
        _ibm_solver->set_t(_t + dt * _dt_ratio);
        _ibm_solver->set_dt(dt * _dt_ratio);

        // 更新高斯积分点当前构型（用于力分布/速度插值）
        _ibm_solver->fun_update_disp();

        // 从流体插值到拉格朗日点得到固体速度
        std::vector<double3> U(_ibm_solver->_solid_displacement.size());
        _ibm_solver->calculate_solid_velocity(_ibm_solver->_solid_displacement, U);

        // 与独立 IBM 显式流程一致：将本步流体速度写回，作为下一步的速度初值
        auto [un, vn, wn, pn] = _fluid_solver->get_velocity_and_pressure();
        (void)pn;
        _ibm_solver->eulerian_velocity_u = algebra::flatten(un);
        _ibm_solver->eulerian_velocity_v = algebra::flatten(vn);
        _ibm_solver->eulerian_velocity_w = algebra::flatten(wn);

        // 显式更新位移：X^{n+1} = X^n + dt * U
        algebra::axpy(dt * _dt_ratio, U, _ibm_solver->_solid_displacement);

        // 同步本类中的状态向量缓存（用于外部监控/兼容）
        _xn_new->resize(_ibm_solver->_solid_displacement_2.size() / 3);
        _xn->resize(_ibm_solver->_solid_displacement_2.size() / 3);
        _xn_1->resize(_ibm_solver->_solid_displacement_2.size() / 3);

        auto disp_flat = algebra::flatten<double3, double>(_ibm_solver->_solid_displacement);
        for (size_t i = 0; i < _xn_new->size(); ++i) {
            _xn_1->data()[i] = _xn->data()[i];
            _xn->data()[i] = _xn_new->data()[i];
            _xn_new->data()[i] = make_double3(disp_flat[3*i], disp_flat[3*i + 1], disp_flat[3*i + 2]);
        }
        
        // ═══════════════════════════════════════════════════════════
        // Step 4: 力学状态反馈到电生理 (力 -> 电)
        // ═══════════════════════════════════════════════════════════
        if (_ep_enabled)
        {
            std::vector<double> lmbda, zetas, zetaw;
            _solid_solver->get_mechanics_feedback(lmbda, zetas, zetaw);
            if (!lmbda.empty()) {
                _ep_solver->update_mechanics_feedback(lmbda, zetas, zetaw);
            } else {
                LOG_F(WARNING, "Mechanics feedback is empty; skip EP update this step.");
            }
        }
        
        // 更新时间
        _t += dt;
        
        LOG_F(INFO, "时间步完成: t_new = %.6e s", _t);
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
        (void)dt;
        const bool ep_active = is_ep_active(t);
        auto XS_vec = ep_active ? _ep_solver->get_XS_field()
                                : std::vector<double>(_solid_mesh->num_dofs(), 0.0);
        auto XW_vec = ep_active ? _ep_solver->get_XW_field()
                                : std::vector<double>(_solid_mesh->num_dofs(), 0.0);
        _solid_solver->set_land_crossbridge_states(XS_vec, XW_vec);
        // Tension is calculated implicitly in solid solver step if implemented
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 访问器
    // ═══════════════════════════════════════════════════════════════════
    
    std::shared_ptr<ElectrophysiologySolver> get_ep_solver() const { return _ep_solver; }
    std::shared_ptr<SolidSolverType> get_solid_solver() const { return _solid_solver; }
    std::shared_ptr<FluidSolverType> get_fluid_solver() const { return _fluid_solver; }
    
    std::shared_ptr<ImmersedBoundaryMethod<FluidSolverType, SolidSolverType>> 
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
        set_output_active_tension(T_func);
        tension_file << *T_func;
    }

    /**
     * 参考 ImmersedBoundaryMethod3D 的输出风格：
     * 同时输出固体与流体 record 数据，并追加 EP 的 Vm 与 Land 主动收缩力 Ta。
     */
    void record()
    {
        LOG_F(INFO, "Record the results.");

        _solid_solver->template record<double3, double>(
            _ibm_solver->_solid_forces, _ibm_solver->_solid_displacement, _t);

        if (_solid_solver->_isoutput) {
            auto vm_func = _ep_solver->get_Vm();
            vm_func->rename("vm", "");
            _solid_solver->file_xdmf->write(*vm_func, _t, XDMFFile::Encoding::HDF5);
            _solid_solver->file_xdmf_checkpoint->write_checkpoint(
                *vm_func, "vm", _t, XDMFFile::Encoding::HDF5, true);

            auto ca_i_func = std::make_shared<Function>(_ep_solver->get_function_space());
            ca_i_func->rename("ca_i", "");
            _ep_solver->set_Ca_i_to_function(ca_i_func);
            _solid_solver->file_xdmf->write(*ca_i_func, _t, XDMFFile::Encoding::HDF5);
            _solid_solver->file_xdmf_checkpoint->write_checkpoint(
                *ca_i_func, "ca_i", _t, XDMFFile::Encoding::HDF5, true);

            auto ta_func = std::make_shared<Function>(_ep_solver->get_function_space());
            ta_func->rename("active_tension", "");
            set_output_active_tension(ta_func);
            _solid_solver->file_xdmf->write(*ta_func, _t, XDMFFile::Encoding::HDF5);
            _solid_solver->file_xdmf_checkpoint->write_checkpoint(
                *ta_func, "active_tension", _t, XDMFFile::Encoding::HDF5, true);
        }

        auto [f1, f2, f3] = _fluid_solver->get_source();
        auto [un, vn, wn, pn] = _fluid_solver->get_velocity_and_pressure();
        _fluid_solver->record(un, vn, wn, f1, f2, f3, pn, _t);

        // 便于直接检查收缩是否生效：输出当前构型体积（与旧 IBM 代码一致的计算方式）
        auto volume_now = _solid_mesh->current_area(_ibm_solver->_solid_displacement);
        LOG_F(WATCH, "Current LV enclosed volume (current_area) = %.16e", volume_now);
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
    std::shared_ptr<ImmersedMeshP1> _solid_mesh;
    std::shared_ptr<BackgroundMesh3D<3>> _fluid_mesh;
    std::shared_ptr<Mesh> _dolfin_mesh;   // = _solid_mesh->get_dolfin_mesh()
    
    // 求解器
    std::shared_ptr<SolidSolverType> _solid_solver;
    std::shared_ptr<FluidSolverType> _fluid_solver;
    std::shared_ptr<ElectrophysiologySolver> _ep_solver;
    std::shared_ptr<ImmersedBoundaryMethod<FluidSolverType, SolidSolverType>> _ibm_solver;
    
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
    bool is_ep_active(double t) const
    {
        if (!_ep_enabled) return false;
        return cycle_time(t) >= _t_end_diastole;
    }

    double cycle_time(double t) const
    {
        if (_t_period <= 0.0) return t;
        double local_t = std::fmod(t, _t_period);
        if (local_t < 0.0) local_t += _t_period;
        return local_t;
    }

    void set_output_active_tension(std::shared_ptr<Function> Ta_func) const
    {
        if (is_ep_active(_t)) {
            _ep_solver->set_land_active_tension_to_function(Ta_func);
        } else {
            zero_function(Ta_func);
        }
    }

    static void zero_function(std::shared_ptr<Function> func)
    {
        std::vector<double> values;
        func->vector()->get_local(values);
        std::fill(values.begin(), values.end(), 0.0);
        func->vector()->set_local(values);
        func->vector()->apply("insert");
    }

    double _t_period;         // 心动周期 [s]

};

} // namespace dolfin

#endif // __ELECTRO_FLUID_STRUCTURE_SOLVER_H__
