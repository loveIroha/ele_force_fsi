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
        self.F = F
        self.u = u
        self.bcs = bcs
        self.J = derivative(F, u)
    
    def F(self, b, x):
        assemble(self.F, tensor=b)
        for bc in self.bcs: bc.apply(b, x)
        
    def J(self, A, x):
        assemble(self.J, tensor=A)
        for bc in self.bcs: bc.apply(A)

# === 形变相关算子 ===
def deformation_operators(U):
    F = Identity(U.geometric_dimension()) + grad(U)  # 变形梯度[^2]
    C = F.T*F                                        # 右Cauchy-Green张量
    J = det(F)
    lambda_ = sqrt(det(C))                           # 拉伸率
    return F, C, lambda_

# === 双域耦合变分方程 ===
def create_variational_form(u, u_old, dt, params):
    Vm, Ca_i, TRPN, U = split(u)
    Vm_old, Ca_i_old, TRPN_old, U_old = split(u_old)
    
    F, C, lambda_ = deformation_operators(U)  # 从参考材料[^2]获取形变量
    
    # 电导方程 (含适配GPB的Ca-TRPN耦合项)
    F_Vm = (params['C_m'] * (Vm - Vm_old)/dt * TestFunction(V.sub(0)) * dx 
            + params['G'] * inner(grad(Vm), grad(TestFunction(V.sub(0)))) * dx 
            - (params['v_stim'] - 0.3*(TRPN - TRPN_old)) * TestFunction(V.sub(0)) * dx)
    
    # TRPN动力学 (参考材料公式[^2])
    Ca_T50_lambda = params['Ca_T50'] * lambda_**2
    dTRPN_dt = ( (1-TRPN)*params['k_TRPN'] * (Ca_i/Ca_T50_lambda)**params['n_TRPN'] 
                - params['k_TRPN']*TRPN )
    F_TRPN = ( (TRPN - TRPN_old)/dt - dTRPN_dt ) * TestFunction(V.sub(2)) * dx
    
    # 钙动态 (含缓冲项)
    F_Ca = ( (Ca_i - Ca_i_old)/dt * TestFunction(V.sub(1)) * dx 
            + (TRPN - TRPN_old)/dt * params['B_TnCL'] * TestFunction(V.sub(1)) * dx )
    
    # 力学平衡方程 (被动+主动应力) [^2]
    S_passive = params['eta'] * (C - Identity(2)) 
    S_active = params['B_TnCL'] * TRPN**3 * Identity(2)  # Land模型主动应力
    S_total = S_passive + S_active
    
    F_mech = inner(S_total, grad(TestFunction(V.sub(3)))) * dx
    
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
      '0.0', '0.0'),       # displacement (U_x, U_y)
     element=V.ufl_element()), V)

u_old = Function(V, name="Solution")
u_new = Function(V, name="Solution_new")
u_old.assign(u_init)

# Boundary condition (固定边界位移)
def boundary(x, on_boundary):
    return on_boundary
bc_D = DirichletBC(V.sub(3), Constant((0,0)), boundary)

# 创建迭代求解器
F = create_variational_form(u_new, u_old, Constant(dt), params)
problem = StronglyCoupledProblem(F, u_new, [bc_D])
solver = NewtonSolver()
solver.parameters["linear_solver"] = "mumps"

vtkfile = File('results/strong_coupling.pvd')

for step in range(num_steps):
    t = step * dt
    
    # 迭代求解耦合系统
    solver.solve(problem, u_new.vector())
    
    # 时步更新
    u_old.assign(u_new)
    
    # 数据输出
    if step % output_freq == 0:
        Vm, Ca_i, TRPN, U = u_new.split(True)
        vtkfile << (Vm, t)
        print(f"Step {step}: Max Vm={Vm.vector().max():.2f}, Avg stretch={project(deformation_operators(U)[2],FunctionSpace(mesh,'P',1)).vector().mean():.3f}")

# 后处理绘制
Vm, Ca_i, TRPN, U = u_new.split()
plt.figure(figsize=(12,4))
plt.subplot(131); plot(Vm); plt.title("Membrane Potential")
plt.subplot(132); plot(project(deformation_operators(U)[2],FunctionSpace(mesh,'P',1))); plt.title("Stretch Ratio λ")
plt.subplot(133); plot(U); plt.title("Displacement Field")
plt.tight_layout()
plt.savefig("final_result.png")
