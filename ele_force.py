from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# === 混合函数空间定义 ===
mesh = UnitSquareMesh(50, 50)
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
mech_element = VectorElement('P', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([P1, P1, P1, mech_element]))  # (Vm, Ca_i, TRPN, U)

# === 建模参数 (来自文献2数据[^2]) ===
params = {
    'C_m': 1.0,         # 膜电容 (μF/cm²)
    'G': 1.17e-3,       # 电导率 (mS/cm)
    'k_TRPN': 0.07,     # TRPN结合速率 (ms⁻¹) [^2]
    'n_TRPN': 3.0,      # 钙敏感性Hill系数[^2]
    'Ca_T50': 0.5e-3,   # 半激活浓度 (mM)
    'B_TnCL': 70.0,     # TnC缓冲容量 (μM) [^2]
    'eta': 5.0e-3,      # 被动刚度 (kPa) 
    'v_stim': 9.5       # 刺激电流幅度 (nA)
}

# === 混合方程的强形式表达 ===
class StronglyCoupledProblem(NonlinearProblem):
    def __init__(self, F, u, bcs):
        super().__init__()
        self.F_form = F
        self.u = u
        self.bcs = bcs
        self.J_form = derivative(F, u)
    
    def F(self, b, x):
        assemble(self.F_form, tensor=b)
        for bc in self.bcs: bc.apply(b, x)
        
    def J(self, A, x):
        assemble(self.J_form, tensor=A)
        for bc in self.bcs: bc.apply(A)

# === 形变相关算子 ===
def deformation_operators(U):
    F = Identity(U.geometric_dimension()) + grad(U)  # 变形梯度[^2]
    C = F.T*F                                        # 右Cauchy-Green张量
    J = det(F)
    lambda_ = sqrt(det(C))                           # 拉伸率
    return F, C, lambda_

# 添加应力计算函数
def compute_stress(U, TRPN, params):
    F, C, lambda_ = deformation_operators(U)
    # 被动应力 (kPa)
    S_passive = params['eta'] * (C - Identity(U.geometric_dimension()))
    # 主动应力 (kPa)
    S_active = params['B_TnCL'] * TRPN**3 * Identity(U.geometric_dimension())
    # 总应力 (kPa)
    S_total = S_passive + S_active
    return S_total

# === 双域耦合变分方程 ===
def create_variational_form(u, u_old, dt, params):
    Vm, Ca_i, TRPN, U = split(u)
    Vm_old, Ca_i_old, TRPN_old, U_old = split(u_old)
    
    # 正确定义测试函数
    v = TestFunction(V)
    v_Vm, v_Ca, v_TRPN, v_U = split(v)
    
    F, C, lambda_ = deformation_operators(U)  # 从参考材料[^2]获取形变量
    
    # 电导方程 (含适配GPB的Ca-TRPN耦合项)
    F_Vm = (params['C_m'] * (Vm - Vm_old)/dt * v_Vm * dx 
            + params['G'] * inner(grad(Vm), grad(v_Vm)) * dx 
            - (params['v_stim'] - 0.3*(TRPN - TRPN_old)) * v_Vm * dx)
    
    # TRPN动力学 (参考材料公式[^2])
    Ca_T50_lambda = params['Ca_T50'] * lambda_**2
    dTRPN_dt = ( (1-TRPN)*params['k_TRPN'] * (Ca_i/Ca_T50_lambda)**params['n_TRPN'] 
                - params['k_TRPN']*TRPN )
    F_TRPN = ( (TRPN - TRPN_old)/dt - dTRPN_dt ) * v_TRPN * dx
    
    # 钙动态 (含缓冲项)
    F_Ca = ( (Ca_i - Ca_i_old)/dt * v_Ca * dx 
            + (TRPN - TRPN_old)/dt * params['B_TnCL'] * v_Ca * dx )
    
    # 力学平衡方程 (被动+主动应力) [^2]
    S_passive = params['eta'] * (C - Identity(2)) 
    S_active = params['B_TnCL'] * TRPN**3 * Identity(2)  # Land模型主动应力
    S_total = S_passive + S_active
    
    F_mech = inner(S_total, grad(v_U)) * dx
    
    return F_Vm + F_TRPN + F_Ca + F_mech

# === 求解器配置 ===
def solver_setup():
    parameters["nonlinear_solver"] = "newton"
    parameters["newton_solver"]["linear_solver"] = "mumps"
    parameters["newton_solver"]["relative_tolerance"] = 1e-6
    parameters["newton_solver"]["report"] = True

# === 主程序 ===
dt = 0.1  # ms
num_steps = 1000
output_freq = 50

# 初始化条件
u_init = interpolate(Expression( 
    ( 'exp(-100*(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)))', # Vm
      '0.0',               # Ca_i (mM) 
      '0.0',               # TRPN
      '0.0', '0.0'),       # displacement (U_x, U_y) - 单位: cm
     element=V.ufl_element()), V)

u_old = Function(V, name="Solution")
u_new = Function(V, name="Solution_new")
u_old.assign(u_init)

# Boundary condition (固定边界位移)
def boundary(x, on_boundary):
    return on_boundary
# 只在左右边界施加零位移约束
def left_right_boundary(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], 1))
bc_D = DirichletBC(V.sub(3), Constant((0,0)), left_right_boundary)

# 创建迭代求解器
F = create_variational_form(u_new, u_old, Constant(dt), params)
problem = StronglyCoupledProblem(F, u_new, [bc_D])
solver = NewtonSolver()
solver.parameters["linear_solver"] = "mumps"

# 创建结果输出目录
import os
if not os.path.exists('2Dresults'):
    os.makedirs('2Dresults')

# 创建输出文件
vtkfile_vm = File('2Dresults/vm.pvd')
vtkfile_displacement = File('2Dresults/displacement_cm.pvd')  # 位移输出 (cm)
vtkfile_stress = File('2Dresults/stress_kpa.pvd')  # 应力输出 (kPa)

# 创建应力函数空间
stress_space = TensorFunctionSpace(mesh, 'P', 1)
stress_function = Function(stress_space, name="Stress_kPa")

for step in range(num_steps):
    t = step * dt
    
    # 迭代求解耦合系统
    solver.solve(problem, u_new.vector())
    
    # 时步更新
    u_old.assign(u_new)
    
    # 数据输出
    if step % output_freq == 0:
        # 使用split(True)获取实际的Function对象而不是UFL表达式
        Vm, Ca_i, TRPN, U = u_new.split(True)
        
        # 输出膜电位
        vtkfile_vm << (Vm, t)
        
        # 输出位移 (cm)
        U.rename("Displacement_cm", "Displacement in cm")
        vtkfile_displacement << (U, t)
        
        # 计算并输出应力 (kPa)
        # 注意：这里需要使用Function对象TRPN而不是UFL表达式
        stress = compute_stress(U, TRPN, params)
        stress_function = project(stress, stress_space)
        stress_function.rename("Stress_kPa", "Stress in kPa")
        vtkfile_stress << (stress_function, t)
        
        # 计算平均拉伸率
        avg_stretch = project(deformation_operators(U)[2], FunctionSpace(mesh, 'P', 1)).vector().get_local().mean()
        
        # 计算最大应力值
        max_stress = stress_function.vector().max()
        
        print(f"Step {step}: Max Vm={Vm.vector().max():.2f} mV, "
              f"Avg stretch={avg_stretch:.3f}, "
              f"Max displacement={U.vector().max():.3f} cm, "
              f"Max stress={max_stress:.3f} kPa")

# 后处理绘制 - 使用split(True)获取实际的Function对象
Vm, Ca_i, TRPN, U = u_new.split(True)

# 计算最终应力
stress = compute_stress(U, TRPN, params)
stress_function = project(stress, stress_space)

plt.figure(figsize=(16,4))
plt.subplot(141)
p1 = plot(Vm)
plt.colorbar(p1)
plt.title("膜电位 (mV)")

plt.subplot(142)
p2 = plot(project(deformation_operators(U)[2], FunctionSpace(mesh, 'P', 1)))
plt.colorbar(p2)
plt.title("拉伸率 λ")

plt.subplot(143)
p3 = plot(U)
plt.colorbar(p3)
plt.title("位移场 (cm)")

plt.subplot(144)
# 计算应力的第一主不变量作为标量可视化
stress_invariant = project(tr(stress_function), FunctionSpace(mesh, 'P', 1))
p4 = plot(stress_invariant)
plt.colorbar(p4)
plt.title("应力 (kPa)")

plt.tight_layout()
plt.savefig("final_result_with_stress.png")

# 保存数值结果到CSV文件
import pandas as pd

# 采样点
num_points = 100
x_points = np.linspace(0, 1, num_points)
y_points = np.linspace(0, 1, num_points)
X, Y = np.meshgrid(x_points, y_points)

# 创建数据框
data = {
    'x': X.flatten(),
    'y': Y.flatten(),
    'displacement_x_cm': np.zeros(num_points**2),
    'displacement_y_cm': np.zeros(num_points**2),
    'stress_kpa': np.zeros(num_points**2)
}

# 创建用于评估的函数空间
V_scalar = FunctionSpace(mesh, 'P', 1)
V_vector = VectorFunctionSpace(mesh, 'P', 1)

# 将位移投影到向量空间以便于评估
U_proj = project(U, V_vector)
# 分离x和y分量
u_x = Function(V_scalar)
u_y = Function(V_scalar)
u_x.vector()[:] = U_proj.vector().get_local()[0::2]  # 提取x分量
u_y.vector()[:] = U_proj.vector().get_local()[1::2]  # 提取y分量

# 计算应力并投影到标量空间
stress_scalar = project(tr(stress_function), V_scalar)

# 填充数据框
for i in range(len(data['x'])):
    point = Point(data['x'][i], data['y'][i])
    try:
        # 使用eval方法在特定点评估函数值
        data['displacement_x_cm'][i] = u_x(point)
        data['displacement_y_cm'][i] = u_y(point)
        data['stress_kpa'][i] = stress_scalar(point)
    except Exception as e:
        print(f"点 ({data['x'][i]}, {data['y'][i]}) 评估失败: {e}")

# 保存到CSV
df = pd.DataFrame(data)
df.to_csv('2Dresults/displacement_and_stress.csv', index=False)

print("模拟完成。位移结果(cm)和应力结果(kPa)已保存到2Dresults目录")
