
"""不加流固耦合程序, Richardson论文中Darcy系统的验证且判断其是否收敛""" 


from fenics import *
from mshr import *
import ufl
import numpy as np
import sympy as sym
from dolfin import *

T = 2
num_steps = 64
dt = T / num_steps
M_b = 2.18e5
b = 1.0
rho = 1e3
kappa = 0.01
phi_0 = 0.1
P0 = 0

n = 32
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'P', 2)
VT = VectorFunctionSpace(mesh, 'P', 1)

S = Constant(0)
K = 1e-7*Identity(2)

#定义边界条件
P_D = Expression('1e4 * (1.0 - exp(-t * t / 0.25))', degree=1, t=0)

#定义边界子域
class Boundary_1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Boundary_2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)

class Boundary_3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], 1.0))

#创建边界子域实例
boundary_1 = Boundary_1()
boundary_2 = Boundary_2()
boundary_3 = Boundary_3()

#创建MeshFunction 用于存储边界标记
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)

#标记边界
boundary_1.mark(boundary_markers, 4)
boundary_2.mark(boundary_markers, 1)
boundary_3.mark(boundary_markers, 2)

File("b.pvd") << boundary_markers

boundary_conditions = { 4: {'Dirichlet': P_D},
                        1: {'Dirichlet':  Expression('0.0',degree=1)},
                        2: {'Neumann': 0}      
}

#边界条件
bcs = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
                         boundary_markers, i)
        bcs.append(bc)

# Define variational problem
P= TrialFunction(V)
Q = TestFunction(V)
M = TrialFunction(V)
N = TestFunction(V)
# P_n = interpolate(P_D, V)
P_n = Function(V)
M_n = Function(V) 
M_n_1 = Function(V)
W = Function(VT)

X = Expression(('0.001*x[0]* (1.0 - exp(-t * t / 0.25))','0.0'), element=VT.ufl_element(), domain= mesh, t=0)
X_1 = Function(VT)
F = ufl.grad(X)+Identity(2)
F1 = ufl.grad(X_1)+Identity(2)
J = ufl.det(F)
J1= ufl.det(F1)
F_1 = ufl.inv(F)
F_T = ufl.transpose(F_1)


def f(J):
    f1 = 1.0
    f2 = 2.0*(J-1.0-ln(J))/((J-1)**2)
    return conditional(le(abs(J-1),1e-6),f1,f2)

def p(M,M_n,J):
    return M_b*(b*(1.0-J) + M/rho)*f(J)-kappa*(rho/(M_n+rho*phi_0))


def grad_f(J):
    f1 = -2/3                                            
    f2 = (2.0*(J-1)**2-4.0*J*(J-1-ln(J)))/(J*(J-1)**3)
    return conditional(le(abs(J-1),1e-4),f1,f2)

def A(J):
    return M_b*f(J)/rho

def B(M_n,J):
    return -M_b*f(J)*b + M_b*grad_f(J)*(b*(1-J)+ M_n/rho)

def C(M_n_1,M_n):
    return (kappa*rho/(M_n + rho*phi_0) - kappa*rho/(M_n_1 + rho*phi_0))/dt

def E(M_n, M_n_1, J, J1):
    return A(J)*rho*S + B(M_n,J)*(J - J1)/dt -C(M_n_1, M_n)

F1 = inner(2*(P-P_n)/dt, Q)*dx + M_b*inner(J*F_1*K*F_T*grad(P),grad(Q*f(J)))*dx-E(M_n, M_n_1, J, J1)*Q*dx
# F1 = inner(Constant(-6.0), Q)*dx + inner(grad(P),grad(Q))*dx
a1 = lhs(F1)
L1 = rhs(F1)
print(a1)
print(L1)

F2 = inner(2*(M-M_n)/dt,N)*dx -rho*S*N*dx + rho*div(W)*N*dx
a2 = lhs(F2)
L2 = rhs(F2)

#Create VTK file for saving solution
vtkfile_p = File('darcy2/aapressure.pvd')
vtkfile_w = File('darcy2/velocity.pvd')
vtkfile_m_1 = File('darcy2/addmass_1.pvd')
vtkfile_m_2 = File('darcy2/addmass_2.pvd')
vtkfile_X = File('darcy2/displacement.pvd')

#Time_stepping
P = Function(V)
M = Function(V)
t = 0
for n in range(num_steps):
    print(t)
    # Update current time
    t += dt/2
    P_D.t = t
    X.t=t
    # Compute 
    #第一步，求解灌注压力
    solve(a1==L1, P, bcs)

    #第二步，求解灌注速度
    W1 = -J*F_1*K*F_T*grad(P)
    W = project(W1, VT)

    #第三步，求解添加质量
    M_1 = rho*(1/(M_b*f(J))*(P - P0 + kappa*(rho/(M_n+rho*phi_0)))+b*(J-1))
    M_1 = project(M_1, V)
    M.assign(M_1)

    vtkfile_m_1  << (M, t)

    # Update previous solution
    t+=dt/2
    print(t)
    P_D.t=t
    P_n.assign(P)
    M_n_1.assign(M_n)
    M_n.assign(M)

    #第四步，求解灌注压力
    solve(a1==L1, P, bcs)

    #第五步，求解灌注速度
    W2 = -J*F_1*K*F_T*grad(P)
    W = project(W2, VT)

    #第六步，求解添加质量
    solve(a2==L2, M)

    #Save to file and plot solution
    vtkfile_p  << (P, t)
    vtkfile_w  << (W, t)
    vtkfile_m_2  << (M, t)
    # Update previous solution
    P_n.assign(P)
    M_n_1.assign(M_n)
    M_n.assign(M)
    X_1.assign(X)
    vtkfile_X  << (X_1, t)
    print(M(0.5,0.5))

print(P(0.5,0.5))
# print(np.sqrt((P(0.5,0.5)-499.99994373241265 )*(P(0.5,0.5)-499.99994373241265)))