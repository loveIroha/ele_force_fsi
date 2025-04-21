from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import max_value, min_value
# === 凹型网格生成 === 
def create_notched_mesh():
    # 定义5x5x5立方体
    mesh = BoxMesh(Point(0,0,0), Point(5,5,5), 5,5,5)
    
    # 定义凹口区域
    class Notch(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] >= 1) & (x[0] <= 4) & (x[1] >= 1) & (x[1] <= 4) & (x[2] >= 1)
    
    # 标记凹口区域
    markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    markers.set_all(0)
    Notch().mark(markers, 1)
    
    # 生成凹型网格 - 使用SubMesh替代MeshView
    submesh = SubMesh(mesh, markers, 0)
    return submesh

mesh = create_notched_mesh()

# === 混合函数空间 ===[^1][^2]
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
vecP1 = VectorElement('P', mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, MixedElement([P1, P1, P1, vecP1]))

V = FunctionSpace(mesh, 'P', 1)
TRPN = Function(V)
TRPN_old = Function(V)

# === 模型参数 ===[^2]
params = {
    'C_m': 1.0,         # μF/cm²
    'G': 1.17e-3,       # mS/cm
    'k_TRPN': 0.14,     # ms⁻¹
    'n_TRPN': 1.54,      # Hill系数
    'Ca_T50': 0.52,   # mM
    'B_TnCL': 70.0,     # μM
    'eta': 0.5,      # kPa
    'v_stim': 9.5       # nA
}

# === 形变相关算子 ===[^2]
def deformation_gradient(U):
    I = Identity(3)
    return I + grad(U)


def compute_stress(U, Ca_i, TRPN):
    F = deformation_gradient(U)
    C = F.T * F
    J = det(F)

    # 被动应力
    S_passive = params['eta'] * (J**(-2/3)*C - Identity(3))

    # 主动张力大小
    T_active = (params['B_TnCL'] * Ca_i**params['n_TRPN'] / (params['Ca_T50'] + Ca_i**params['n_TRPN'])) * TRPN**3

    S_active = T_active / lambda_f * outer(fiber, fiber)

    return S_passive + S_active

# === 变分形式 ===[^1][^2]
u = Function(ME)
u_old = Function(ME)
Vm, Ca_i, TRPN, U = split(u)
Vm_old, Ca_i_old, TRPN_old, U_old = split(u_old)

# 时间参数
dt = 0.1
t = 0.0
T = 650

# 测试函数
w1, w2, w3, w4 = TestFunctions(ME)

# 电生理方程
def I_ion(Vm, Ca_i):
    # 改为更温和的限制函数（避免爆炸）
    return -0.05 * Vm + 0.001 * Ca_i



F_Vm = (
    params['C_m'] * (Vm - Vm_old)/dt * w1 * dx
    + inner(params['G'] * grad(Vm), grad(w1)) * dx
    - params['v_stim'] * w1 * dx
    + I_ion(Vm, Ca_i) * w1 * dx                      # 加入Vm自限机制
    - 0.3 * (TRPN - TRPN_old) * w1 * dx   # 保留TRPN耦合项
)
# F_Vm = (params['C_m'] * (Vm - Vm_old)/dt * w1 * dx 
#        + inner(params['G'] * grad(Vm), grad(w1)) * dx 
#        - params['v_stim'] * w1 * dx 
#        - 0.3 * (TRPN - TRPN_old) * w1 * dx)  # TRPN耦合项[^1]

# TRPN动力学[^2]
F_ = deformation_gradient(U)
C_ = F_.T*F_
fiber = Constant((1.0, 0.0, 0.0))  # 纤维方向
lambda_f = sqrt(inner(fiber, C_ * fiber))
Ca_T50_lambda = params['Ca_T50'] * lambda_f**2
# dTRPN_dt = params['k_TRPN'] * (Ca_i**params['n_TRPN'] / (Ca_T50_lambda + Ca_i**params['n_TRPN']) * (1 - TRPN) - TRPN)
gamma = Ca_i**params['n_TRPN'] / (Ca_T50_lambda + Ca_i**params['n_TRPN'])
dTRPN_dt = params['k_TRPN'] * (gamma * (1 - TRPN) - TRPN)
F_TRPN = (TRPN - TRPN_old)/dt * w3 * dx - dTRPN_dt * w3 * dx

# 钙缓冲动力学[^1]
F_Ca = (Ca_i - Ca_i_old)/dt * w2 * dx + (TRPN - TRPN_old)/dt * params['B_TnCL'] * w2 * dx

# 力学平衡方程
S_total = compute_stress(U,Ca_i, TRPN)
F_mech = inner(S_total, grad(w4)) * dx

# 总变分形式
F = F_Vm + F_TRPN + F_Ca + F_mech

# === 边界条件 ===
# 力学边界条件 - 固定外部边界
def mechanical_boundary(x, on_boundary):
    return on_boundary and not (near(x[0],0) or near(x[1],0) or near(x[2],0))
bc_u = DirichletBC(ME.sub(3), Constant((0,0,0)), mechanical_boundary)


# === 初始条件 ===
class InitialCondition(UserExpression):
    def eval(self, values, x):
        # 定义内膜表面 - 近似为立方体的左右两侧内表面
        is_endocardium = near(x[0], 1.0) or near(x[0], 4.0)
        
        # 在内膜表面同步激发，其他区域为静息电位
        values[0] = 10.0 if  near(x[0], 1.0) else -85.0  # Vm: 内膜表面为10mV，其他区域为静息电位-85mV
        values[1] = 0.107                        # Ca_i: 设置为健康心肌细胞的基线值
        values[2] = 0.1                              # TRPN  
        values[3] = 0.0                               # Ux
        values[4] = 0.0                               # Uy 
        values[5] = 0.0                               # Uz
        
    def value_shape(self): return (6,)

u_init = InitialCondition(degree=2)
u.assign(interpolate(u_init, ME))
u_old.assign(u)

# === 求解器设置 ===

problem = NonlinearVariationalProblem(F, u, bcs=[bc_u], J=derivative(F, u))
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6

# === 结果输出 ===
# 只使用PVD文件输出
vm_pvd = File("3Dresults/Vm.pvd")
ca_pvd = File("3Dresults/Ca.pvd")
trpn_pvd = File("3Dresults/TRPN.pvd")
stress_pvd = File("3Dresults/Stress.pvd")
disp_pvd = File("3Dresults/Displacement.pvd")

step = 0
while t <= T:
    solver.solve()
    
    # 保存结果
    if step % 10 == 0:
        # 分离解
        _Vm, _Ca, _TRPN, _U = u.split()
        
        # 将分离的解投影到单独的函数空间以便正确保存
        Vm_proj = project(_Vm, FunctionSpace(mesh, "P", 1))
        Ca_proj = project(_Ca, FunctionSpace(mesh, "P", 1))
        TRPN_proj = project(_TRPN, FunctionSpace(mesh, "P", 1))
        U_proj = project(_U, VectorFunctionSpace(mesh, "P", 1))
        
        # 计算主动应力
        S = project(compute_stress(_U, _Ca, _TRPN), TensorFunctionSpace(mesh, "P", 1))
        
        # 保存PVD格式
        vm_pvd << (Vm_proj, t)
        ca_pvd << (Ca_proj, t)
        trpn_pvd << (TRPN_proj, t)
        stress_pvd << (S, t)
        disp_pvd << (U_proj, t)
        
        print(f"Time={t:.1f}ms, Max Vm={Vm_proj.vector().max():.2f}, Stress Norm={S.vector().norm('l2'):.2e}")

    u_old.assign(u)
    t += dt
    step += 1

# === 后处理可视化 ===
# 使用最后一个时间步的数据进行可视化
# 创建新的函数用于可视化
Vm_viz = project(_Vm, FunctionSpace(mesh, "P", 1))
Stress_viz = project(compute_stress(_U, _TRPN), TensorFunctionSpace(mesh, "P", 1))
U_viz = project(_U, VectorFunctionSpace(mesh, "P", 1))

# 绘图并保存为PNG格式
plt.figure(figsize=(15,5))

plt.subplot(131)
p1 = plot(Vm_viz, title="Membrane Potential")
plt.colorbar(p1)

plt.subplot(132)
p2 = plot(Stress_viz[0,0], title="Stress_xx Component")
plt.colorbar(p2)

plt.subplot(133)
p3 = plot(U_viz.sub(2), title="Displacement Z-component")
plt.colorbar(p3)

# 确保保存为PNG格式
plt.tight_layout()
plt.savefig("3Dresults/final_result.png", dpi=300, format='png')
print("后处理完成, 结果已保存至3Dresults/final_result.png")
