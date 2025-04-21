from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# 创建网格
mesh = UnitSquareMesh(50, 50)  # 50x50的单位正方形网格

# 定义函数空间 (P1元)
V = FunctionSpace(mesh, 'P', 1)

# 定义时间步长和总时间
T = 1.0         # 总时间
num_steps = 1000  # 时间步数
dt = T / num_steps  # 时间步长

# 定义方程参数
C_m = Constant(1.0)    # 膜电容
G = Constant(1.17e-3)      # 电导率张量 (这里简化为标量)
I_ion = Constant(0.0)  # 离子电流 (设为常数)

# 定义边界条件
def boundary(x, on_boundary):
    return on_boundary

# 在边界上设置电位为0
bc = DirichletBC(V, Constant(0.0), boundary)

# 定义初始条件 - 在中心区域设置一个电位刺激
v_init = Expression('exp(-100*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))', degree=2)
v_old = interpolate(v_init, V)  # 上一时间步的解

# 定义试验函数和测试函数
v = TrialFunction(V)
w = TestFunction(V)

# 使用向后欧拉方法的变分形式
F = (C_m/dt * inner(v, w) * dx + 
     inner(G * grad(v), grad(w)) * dx + 
     I_ion * w * dx - 
     C_m/dt * inner(v_old, w) * dx)

a, L = lhs(F), rhs(F)

# 创建函数来存储解
v_new = Function(V)  # 当前时间步的解

# 创建VTK文件用于保存结果
vtkfile = File('/home/gjh/data/eleresults/membrane_potential.pvd')

# 保存初始条件
vtkfile << v_old

# 时间迭代求解
t = 0
for n in range(num_steps):
    # 更新时间
    t += dt
    
    # 求解变分问题
    solve(a == L, v_new, bc)
    
    # 更新解
    v_old.assign(v_new)
    
    # 每10步保存一次结果
    if n % 10 == 0:
        vtkfile << (v_new, t)
        print(f'时间 t = {t:.3f}, 最大电位 = {v_new.vector().max():.3f}')

print('计算完成!')

# 绘制最终结果
p = plot(v_new)
plt.colorbar(p)
plt.title('膜电位分布 (t = %.2f)' % T)
plt.savefig('/home/gjh/data/eleresults/final_potential.png')
plt.show()