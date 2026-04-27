/**
 * @file GPBTissueManager.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief GPB 细胞模型组织管理器 - 电-流-固耦合框架
 * @version 0.2
 * @date 2026-03-09
 * 
 * @copyright Copyright (c) 2026 Ma Pengfei
 * 
 * 适配 Mesh_test 流固耦合框架，支持：
 * - 多尺度时间步进（ODE子循环）
 * - 与固体求解器的力学状态交互（λ, dλ/dt）
 * - MPI 并行 + OpenMP 共享内存混合并行
 * - MeshFunction 单元级数据交换
 */

#ifndef __GPB_TISSUE_MANAGER_H__
#define __GPB_TISSUE_MANAGER_H__

#include <dolfin.h>
#include <mpi.h>
#include <vector>
#include <memory>
#include <omp.h>
#include <loguru/loguru.hpp>

#include "GPB_cell_model.h"
#include "ActiveContractionGPB.h"

namespace dolfin {

/**
 * GPBTissueManager - Multi-scale time stepping with subcycling
 * 
 * Time scale separation strategy:
 * - PDE (tissue): Large time step (dt_PDE ~ 0.05-0.1 ms)
 * - ODE (cell):   Small time steps (dt_ODE ~ 0.01 ms)
 * - Subcycling:   Multiple ODE steps per PDE step
 * 
 * Operator splitting with subcycling:
 *   For each PDE step Δt_PDE:
 *     1. Fix Vm at current value
 *     2. Take N_sub ODE substeps: dt_ODE = Δt_PDE / N_sub
 *     3. Average I_ion over substeps
 *     4. Use averaged I_ion for PDE step to get Vm^{n+1}
 * 
 * 与流固耦合集成:
 *   - 从固体求解器获取变形梯度(lambda, dlambda_dt)
 *   - 输出主动收缩张力到固体求解器
 *   - 支持 MeshFunction<double> 单元级数据交换
 */
class GPBTissueManager
{
public:
    /**
     * Constructor with subcycling support
     * 
     * @param function_space  FEniCS scalar function space (节点级)
     * @param mesh            FEniCS mesh (用于单元级数据)
     * @param dt_pde_ms      PDE time step (milliseconds) - larger
     * @param dt_ode_ms      ODE time step (milliseconds) - smaller
     */
    GPBTissueManager(
        std::shared_ptr<FunctionSpace> function_space,
        std::shared_ptr<Mesh> mesh,
        double dt_pde_ms,
        double dt_ode_ms)
        : V_space(function_space),
          _mesh(mesh),
          dt_pde_milliseconds(dt_pde_ms),
          dt_ode_milliseconds(dt_ode_ms)
    {
        LOG_F(INFO, "初始化 GPB 组织管理器 (电-流-固耦合)");
        
        // Calculate number of ODE substeps per PDE step
        n_substeps = static_cast<int>(std::round(dt_pde_ms / dt_ode_ms));
        
        // Ensure at least 1 substep
        if (n_substeps < 1) {
            LOG_F(WARNING, "dt_ODE > dt_PDE, setting n_substeps = 1");
            n_substeps = 1;
            dt_ode_milliseconds = dt_pde_milliseconds;
        }
        
        // Adjust dt_ode to exactly divide dt_pde
        dt_ode_milliseconds = dt_pde_milliseconds / n_substeps;
        
        // MPI并行：获取本地自由度数（而非全局）
        auto dof_range = V_space->dofmap()->ownership_range();
        num_nodes_local = dof_range.second - dof_range.first;
        num_nodes_global = V_space->dim();
        num_cells = _mesh->num_cells();
        
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "GPBTissueManager 初始化参数:");
        LOG_F(INFO, "  本地节点数: %zu / 全局: %zu", num_nodes_local, num_nodes_global);
        LOG_F(INFO, "  单元数: %zu", num_cells);
        LOG_F(INFO, "  每节点状态数: %d", GPBCellModel::CELL_STATE_DIM);
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_F(INFO, "时间尺度分离策略:");
        LOG_F(INFO, "  PDE 时间步长 (Δt_tissue): %.4f ms", dt_pde_milliseconds);
        LOG_F(INFO, "  ODE 时间步长 (Δt_cell):   %.4f ms", dt_ode_milliseconds);
        LOG_F(INFO, "  子循环步数 (N_substeps):  %d", n_substeps);
        LOG_F(INFO, "  加速比 (理论):            %dx", n_substeps);
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Initialize node-level cell states (使用本地节点数)
        cell_states.resize(num_nodes_local);
        std::vector<double> resting_state = GPBCellModel::get_initial_state();
        
        for (size_t i = 0; i < num_nodes_local; ++i) {
            cell_states[i] = resting_state;
        }
        
        // Initialize node-level NHS states and tension
        nhs_states.resize(num_nodes_local);
        active_tension.resize(num_nodes_local, 0.0);
        
        // Initialize mechanical state arrays (节点级)
        lambda_field.resize(num_nodes_local, 1.0);      // 纤维方向拉伸比
        dlambda_dt_field.resize(num_nodes_local, 0.0);  // 拉伸比变化率
        
        // Initialize cell-level MeshFunctions (单元级，用于与固体求解器交互)
        _T_cell = std::make_shared<MeshFunction<double>>(_mesh, 3, 0.0);
        _Ca_i_cell = std::make_shared<MeshFunction<double>>(_mesh, 3, 0.0);
        
        LOG_F(INFO, "  所有细胞已初始化为静息态");
        LOG_F(INFO, "  NHS active contraction model enabled");
        LOG_F(INFO, "  力学耦合接口已初始化 (lambda, dlambda_dt)");
        LOG_F(INFO, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
    
    /**
     * 简化构造函数（向后兼容）
     */
    GPBTissueManager(
        std::shared_ptr<FunctionSpace> function_space, 
        double dt_pde_ms,
        double dt_ode_ms)
        : GPBTissueManager(function_space, 
                           std::const_pointer_cast<Mesh>(function_space->mesh()),
                           dt_pde_ms, dt_ode_ms)
    {}
    
    /**
     * Compute ionic current with subcycling
     * 
     * Algorithm:
     * 1. Hold Vm constant at current value
     * 2. Take n_substeps ODE substeps with dt_ode
     * 3. Average I_ion over all substeps
     * 4. Return averaged I_ion for PDE
     * 
     * @param Vm_function     Current membrane potential (V)
     * @param I_ion_function  Output averaged ionic current (μA/cm²)
     * @param time_ms         Current time (milliseconds)
     */
    void compute_ionic_current_subcycling(
        std::shared_ptr<Function> Vm_function,
        std::shared_ptr<Function> I_ion_function,
        double time_ms)
    {
        // Get Vm array from FEniCS (本地数据)
        std::vector<double> Vm_array;
        Vm_function->vector()->get_local(Vm_array);
        
        if (Vm_array.size() != num_nodes_local) {
            throw std::runtime_error("Vm array size mismatch! Expected: " 
                + std::to_string(num_nodes_local) + ", Got: " 
                + std::to_string(Vm_array.size()));
        }
        
        // Prepare output array for averaged I_ion
        std::vector<double> I_ion_avg(num_nodes_local, 0.0);
        
        // ═══════════════════════════════════════════════════
        // SUBCYCLING LOOP: Take multiple ODE substeps
        // ═══════════════════════════════════════════════════
        for (int substep = 0; substep < n_substeps; ++substep) {
            
            double t_sub = time_ms + substep * dt_ode_milliseconds;
            
            // ┌─────────────────────────────────────────────────┐
            // │ OpenMP 并行化：每个细胞独立计算                 │
            // └─────────────────────────────────────────────────┘
            #pragma omp parallel for schedule(static)
            for (size_t node = 0; node < num_nodes_local; ++node) {
                
                // Convert V → mV
                double Vm_volts = Vm_array[node];
                double Vm_millivolts = Vm_volts * 1000.0;
                
                // Call GPB cell model
                GPBCellModel::Result result = GPBCellModel::compute_ionic_current(
                    t_sub,
                    cell_states[node],
                    Vm_millivolts
                );
                
                // Check validity
                if (std::isnan(result.I_ion) || std::isinf(result.I_ion)) {
                    // 不在并行区域打印，避免 I/O 竞争
                    result.I_ion = 0.0;
                }
                
                // Accumulate I_ion for averaging
                I_ion_avg[node] += result.I_ion;
                
                // Update cell states using Forward Euler
                for (size_t j = 0; j < GPBCellModel::CELL_STATE_DIM; ++j) {
                    cell_states[node][j] += dt_ode_milliseconds * result.ydot[j];
                    
                    // Check state validity
                    if (std::isnan(cell_states[node][j]) || std::isinf(cell_states[node][j])) {
                        cell_states[node][j] = 0.0;
                    }
                }
                
                // NHS active contraction model
                double Ca_i_mM = cell_states[node][37];
                
                // 使用力学状态更新 NHS 主动收缩模型
                // lambda: 纤维方向拉伸比, dlambda_dt: 拉伸比变化率
                double lambda = lambda_field[node];
                double dlambda_dt = dlambda_dt_field[node];
                
                ActiveContractionNHS::update_nhs_state(
                    nhs_states[node],
                    Ca_i_mM,
                    lambda,      // 从固体求解器获取
                    dlambda_dt,  // 从固体求解器获取
                    dt_ode_milliseconds
                );
                active_tension[node] = ActiveContractionNHS::compute_active_tension(
                    nhs_states[node],
                    lambda,
                    dlambda_dt
                );
            }
        }
        
        // ═══════════════════════════════════════════════════
        // Average I_ion over all substeps (也可以并行化)
        // ═══════════════════════════════════════════════════
        #pragma omp parallel for simd
        for (size_t node = 0; node < num_nodes_local; ++node) {
            I_ion_avg[node] /= n_substeps;
        }
        
        // Write averaged I_ion back to FEniCS
        I_ion_function->vector()->set_local(I_ion_avg);
        I_ion_function->vector()->apply("insert");
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // 与固体求解器的力学耦合接口
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 设置纤维方向拉伸比场（从固体求解器获取）
     * @param lambda_values  节点级拉伸比数组
     */
    void set_lambda_field(const std::vector<double>& lambda_values) {
        if (lambda_values.size() != num_nodes_local) {
            LOG_F(WARNING, "lambda_field size mismatch: expected %zu, got %zu",
                  num_nodes_local, lambda_values.size());
            return;
        }
        lambda_field = lambda_values;
    }
    
    /**
     * 设置拉伸比变化率场（从固体求解器获取）
     * @param dlambda_dt_values  节点级拉伸比变化率数组
     */
    void set_dlambda_dt_field(const std::vector<double>& dlambda_dt_values) {
        if (dlambda_dt_values.size() != num_nodes_local) {
            LOG_F(WARNING, "dlambda_dt_field size mismatch: expected %zu, got %zu",
                  num_nodes_local, dlambda_dt_values.size());
            return;
        }
        dlambda_dt_field = dlambda_dt_values;
    }
    
    /**
     * 一次性设置力学状态（更高效）
     */
    void set_mechanical_state(const std::vector<double>& lambda_values,
                              const std::vector<double>& dlambda_dt_values) {
        set_lambda_field(lambda_values);
        set_dlambda_dt_field(dlambda_dt_values);
    }
    
    /**
     * 获取纤维方向拉伸比场
     */
    const std::vector<double>& get_lambda_field() const { return lambda_field; }
    
    /**
     * 获取拉伸比变化率场
     */
    const std::vector<double>& get_dlambda_dt_field() const { return dlambda_dt_field; }
    
    // ═══════════════════════════════════════════════════════════════════
    // 单元级数据接口（与 ActiveLeftVentricleSolver 交互）
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * 获取单元级主动张力 MeshFunction (用于 UFL 形式)
     */
    std::shared_ptr<MeshFunction<double>> get_T_cell_meshfunction() const {
        return _T_cell;
    }
    
    /**
     * 获取单元级钙离子浓度 MeshFunction
     */
    std::shared_ptr<MeshFunction<double>> get_Ca_i_cell_meshfunction() const {
        return _Ca_i_cell;
    }
    
    /**
     * 将节点级张力插值到单元级（用于固体求解器）
     * 简单取单元所有节点的平均值
     */
    void interpolate_tension_to_cells() {
        LOG_SCOPE_FUNCTION(INFO);
        
        if (!_mesh) {
            LOG_F(WARNING, "Mesh not set, cannot interpolate to cells");
            return;
        }
        
        const auto& dofmap = *V_space->dofmap();
        
        // cell_dofs() 返回本地 DOF 索引 [0, num_local + num_ghost)
        // 本进程拥有的 DOF 索引在 [0, num_nodes_local) 范围内
        for (CellIterator cell(*_mesh); !cell.end(); ++cell) {
            auto cell_dofs = dofmap.cell_dofs(cell->index());
            double T_sum = 0.0;
            double Ca_i_sum = 0.0;
            int count = 0;
            
            for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
                auto local_idx = cell_dofs[i];
                // 检查是否为本地拥有的自由度（非 ghost）
                if (local_idx >= 0 && static_cast<size_t>(local_idx) < num_nodes_local) {
                    T_sum += active_tension[local_idx];
                    Ca_i_sum += cell_states[local_idx][37] * 1000.0;  // mM -> μM
                    count++;
                }
            }
            
            if (count > 0) {
                (*_T_cell)[cell->index()] = T_sum / count;
                (*_Ca_i_cell)[cell->index()] = Ca_i_sum / count;
            }
        }
    }
    
    /**
     * Original single-step method (for backward compatibility)
     */
    void compute_ionic_current(
        std::shared_ptr<Function> Vm_function,
        std::shared_ptr<Function> I_ion_function,
        double time_ms)
    {
        // Just call subcycling version
        compute_ionic_current_subcycling(Vm_function, I_ion_function, time_ms);
    }
    
    // Getters for time step info
    double get_dt_pde_ms() const { return dt_pde_milliseconds; }
    double get_dt_ode_ms() const { return dt_ode_milliseconds; }
    int get_n_substeps() const { return n_substeps; }
    
    // Getter for mesh
    std::shared_ptr<Mesh> get_mesh() const { return _mesh; }
    
    /**
     * Get statistics
     */
    struct Statistics {
        double Vm_min_mV;
        double Vm_max_mV;
        double Vm_mean_mV;
        double I_ion_min;
        double I_ion_max;
        double I_ion_mean;
        double Ca_i_min_mM;
        double Ca_i_max_mM;
        double Ca_i_mean_mM;
        size_t num_nodes;
    };
    
    Statistics get_statistics(
        std::shared_ptr<Function> Vm_function,
        std::shared_ptr<Function> I_ion_function) const
    {
        Statistics stats;
        stats.num_nodes = num_nodes_global;
        
        // FEniCS的min/max/sum已经是全局归约后的结果
        stats.Vm_min_mV = Vm_function->vector()->min() * 1000.0;
        stats.Vm_max_mV = Vm_function->vector()->max() * 1000.0;
        stats.Vm_mean_mV = Vm_function->vector()->sum() / num_nodes_global * 1000.0;
        
        stats.I_ion_min = I_ion_function->vector()->min();
        stats.I_ion_max = I_ion_function->vector()->max();
        stats.I_ion_mean = I_ion_function->vector()->sum() / num_nodes_global;

        double Ca_i_sum = 0.0;
        double Ca_i_max = -1e9;
        double Ca_i_min = 1e9;
        
        for (size_t node = 0; node < num_nodes_local; ++node) {
            // 乘以1000转换量纲：从 mM 转为 μM
            double Ca_i = cell_states[node][37] * 1000.0;
            Ca_i_min = std::min(Ca_i_min, Ca_i);
            Ca_i_max = std::max(Ca_i_max, Ca_i);
            Ca_i_sum += Ca_i;
        }
        
        // MPI归约获取全局Ca_i统计
        double global_Ca_i_min, global_Ca_i_max, global_Ca_i_sum;
        MPI_Allreduce(&Ca_i_min, &global_Ca_i_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&Ca_i_max, &global_Ca_i_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&Ca_i_sum, &global_Ca_i_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
        stats.Ca_i_min_mM = global_Ca_i_min;
        stats.Ca_i_max_mM = global_Ca_i_max;
        stats.Ca_i_mean_mM = global_Ca_i_sum / num_nodes_global;
        return stats;
    }

    std::vector<double> get_Ca_i_vector() const
    {
        std::vector<double> Ca_i_values(num_nodes_local);
        
        for (size_t node = 0; node < num_nodes_local; ++node) {
            Ca_i_values[node] = cell_states[node][37] * 1000.0;
        }
        
        return Ca_i_values;
    }
    
    bool check_stability(
        std::shared_ptr<Function> Vm_function,
        std::shared_ptr<Function> I_ion_function,
        double acceptable_Vm_range_mV = 200.0) const
    {
        double Vm_min = Vm_function->vector()->min() * 1000.0;
        double Vm_max = Vm_function->vector()->max() * 1000.0;
        
        if (std::isnan(Vm_min) || std::isnan(Vm_max) ||
            std::isinf(Vm_min) || std::isinf(Vm_max)) {
            LOG_F(ERROR, "Vm contains NaN or Inf!");
            return false;
        }
        
        if (Vm_max > acceptable_Vm_range_mV || Vm_min < -acceptable_Vm_range_mV) {
            LOG_F(WARNING, "Vm out of range [%.2f, %.2f] mV", Vm_min, Vm_max);
            return false;
        }
        
        double I_ion_min = I_ion_function->vector()->min();
        double I_ion_max = I_ion_function->vector()->max();
        
        if (std::isnan(I_ion_min) || std::isnan(I_ion_max) ||
            std::isinf(I_ion_min) || std::isinf(I_ion_max)) {
            LOG_F(ERROR, "I_ion contains NaN or Inf!");
            return false;
        }
        
        return true;
    }
    
    const std::vector<double>& get_cell_state(size_t node_index) const {
        if (node_index >= num_nodes_local) {
            throw std::out_of_range("Node index out of range");
        }
        return cell_states[node_index];
    }
    
    void print_debug_info(size_t node_index = 0) const {
        if (node_index >= num_nodes_local) return;
        
        // 状态索引 (组织模式 57 维, 不含 Vm):
        //   [31]=Na_junc, [32]=Na_sl, [33]=Na_i
        //   [34]=K_i
        //   [35]=Ca_junc, [36]=Ca_sl, [37]=Ca_i
        LOG_F(INFO, "Node %zu cell state summary:", node_index);
        LOG_F(INFO, "  Na_i: %.4f mM", cell_states[node_index][33]);
        LOG_F(INFO, "  K_i: %.4f mM", cell_states[node_index][34]);
        LOG_F(INFO, "  Ca_i: %.4f μM", cell_states[node_index][37] * 1000.0);
        LOG_F(INFO, "  lambda: %.4f", lambda_field[node_index]);
        LOG_F(INFO, "  dlambda_dt: %.4f", dlambda_dt_field[node_index]);
        LOG_F(INFO, "  Active_Tension: %.4f kPa", active_tension[node_index]);
    }

    /**
     * Get active tension field (kPa)
     */
    const std::vector<double>& get_active_tension() const {
        return active_tension;
    }
    
    /**
     * Get active tension statistics
     */
    struct TensionStats {
        double T_min_kPa;
        double T_max_kPa;
        double T_mean_kPa;
        double Ca_b_mean_uM;
        double z_mean;
    };
    
    TensionStats get_tension_statistics() const {
        TensionStats stats;
        double T_min = 1e100, T_max = -1e100, T_sum = 0.0;
        double Ca_b_sum = 0.0, z_sum = 0.0;
        
        for (size_t i = 0; i < num_nodes_local; ++i) {
            double T = active_tension[i];
            T_min = std::min(T_min, T);
            T_max = std::max(T_max, T);
            T_sum += T;
            Ca_b_sum += nhs_states[i].Ca_b;
            z_sum += nhs_states[i].z;
        }
        
        // MPI归约获取全局统计
        double global_T_min, global_T_max, global_T_sum;
        double global_Ca_b_sum, global_z_sum;
        MPI_Allreduce(&T_min, &global_T_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&T_max, &global_T_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&T_sum, &global_T_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&Ca_b_sum, &global_Ca_b_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&z_sum, &global_z_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        stats.T_min_kPa = global_T_min;
        stats.T_max_kPa = global_T_max;
        stats.T_mean_kPa = global_T_sum / num_nodes_global;
        stats.Ca_b_mean_uM = global_Ca_b_sum / num_nodes_global;
        stats.z_mean = global_z_sum / num_nodes_global;
        
        return stats;
    }
    
    /**
     * Set active tension to FEniCS function for output/visualization
     */
    void set_tension_function(std::shared_ptr<Function> T_function) {
        T_function->vector()->set_local(active_tension);
        T_function->vector()->apply("insert");
    }

private:
    // ═══════════════════════════════════════════════════════════════════
    // FEniCS 对象
    // ═══════════════════════════════════════════════════════════════════
    std::shared_ptr<FunctionSpace> V_space;
    std::shared_ptr<Mesh> _mesh;
    
    // ═══════════════════════════════════════════════════════════════════
    // 并行信息
    // ═══════════════════════════════════════════════════════════════════
    size_t num_nodes_local;   // 当前进程拥有的本地自由度数
    size_t num_nodes_global;  // 全局总自由度数
    size_t num_cells;         // 网格单元数
    
    // ═══════════════════════════════════════════════════════════════════
    // Time step parameters
    // ═══════════════════════════════════════════════════════════════════
    double dt_pde_milliseconds;   // Large time step for PDE
    double dt_ode_milliseconds;   // Small time step for ODE
    int n_substeps;               // Number of ODE substeps per PDE step
    
    // ═══════════════════════════════════════════════════════════════════
    // 节点级数据 (Node-level data)
    // ═══════════════════════════════════════════════════════════════════
    std::vector<std::vector<double>> cell_states;           // GPB 细胞状态
    std::vector<ActiveContractionNHS::NHSState> nhs_states; // NHS 主动收缩状态
    std::vector<double> active_tension;                     // 主动张力 (kPa)
    
    // ═══════════════════════════════════════════════════════════════════
    // 力学耦合数据 (从固体求解器接收)
    // ═══════════════════════════════════════════════════════════════════
    std::vector<double> lambda_field;       // 纤维方向拉伸比 λ
    std::vector<double> dlambda_dt_field;   // 拉伸比变化率 dλ/dt
    
    // ═══════════════════════════════════════════════════════════════════
    // 单元级数据 (Cell-level data, 与固体求解器交互)
    // ═══════════════════════════════════════════════════════════════════
    std::shared_ptr<MeshFunction<double>> _T_cell;      // 单元级主动张力
    std::shared_ptr<MeshFunction<double>> _Ca_i_cell;   // 单元级钙离子浓度
};

} // namespace dolfin

#endif // __GPB_TISSUE_MANAGER_H__