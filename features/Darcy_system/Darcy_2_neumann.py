
#位移是变量，将边界条件改成neumann边界，验证Darcy方程组可以求解且是正确的


from fenics import *
from mshr import *
import ufl
import numpy as np
import sympy as sym
from dolfin import *

T = 1
num_steps = 512
dt = T / num_steps
M_b = 2.0
b = 1.0
rho = 1.0
kappa = 1.0
phi_0 = 1.0

n = 16
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'P', 1)
VT = VectorFunctionSpace(mesh, 'P', 1)

#S = Expression('4.0*pow(t, 2)*pow(x[0], 2)/pow(t*(1.0 - pow(x[0], 2)) + 1.0, 3) + 1.2274112777602189*t + 1.0*t/pow(t*(1.0 - pow(x[0], 2)) + 1.0, 2) - pow(x[0], 2) + 1', degree=4, t=0)
S = Expression('4.0*pow(t, 2)*pow(x[0], 2)*pow(1 - t, 2)/pow(t*(1 - t)*(1.0 - pow(x[0], 2)) + 1.0, 3) + 1.2274112777602189*t*(1 - t) + 1.0*t*(1 - t)/pow(t*(1 - t)*(1.0 - pow(x[0], 2)) + 1.0, 2) - t*(1.0 - pow(x[0], 2)) + (1 - t)*(1.0 - pow(x[0], 2))', degree = 4, t=0)
K = 1.0*Identity(2)

#Define boundary condition
M_D = Expression('(1.0 -x[0]*x[0])*(1-t)*t', degree=2, t=0)

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

boundary_conditions = { 4: {'Dirichlet': M_D},
                        1: {'Dirichlet': M_D},
                        2: {'Neumann': M_D}      
}

#边界条件
bcs = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
                         boundary_markers, i)
        bcs.append(bc)

ds = Measure('ds', domain = mesh, subdomain_data=boundary_markers)

# Define variational problem
M = TrialFunction(V)
N = TestFunction(V)
M_n = interpolate(M_D, V)
n = FacetNormal(mesh)

# x, y= sym.symbols('x[0], x[1]')
X = Expression(('2*x[0]','x[1]'), element=VT.ufl_element(), domain= mesh)
F = ufl.grad(X)
J = ufl.det(F)
F_1 = ufl.inv(F)
F_T = ufl.transpose(F_1)

def f(J):
    return 2*(J-1-ufl.ln(J))/(J-1)*(J-1)

def p(M,M_n,J):
    return M_b*(b*(1-J) + M/rho)*f(J)-kappa*(rho/(M_n+rho*phi_0))

def grad_p(J, M, M_n):
    #return -M_b*b*f(J)*grad(J) + M_b*b*(1-J)*ufl.grad(f(J)) + M_b/rho*f(J)*grad(M_n) + M_b*M_n/rho*grad(f(J)) + kappa*rho/((M_n+rho*phi_0)**2)*grad(M_n)
    return -M_b*b*f(J)*grad(J) + M_b*b*(1-J)*ufl.grad(f(J)) + M_b*M_n/rho*grad(f(J))
    # return -M_b*b*f(J)*grad(J) + M_b*b*(1-J)*ufl.grad(f(J)) + M_b*M/rho*grad(f(J))

F = (M - M_n)/dt*N*dx - dot(J*F_1*K*F_T*grad_p( J, M, M_n), n)*N*ds(2) + inner((J*F_1*K*F_T*grad(p(M, M_n, J))),grad(N))*dx - S*N*dx
a = lhs(F)
L = rhs(F)

#Create VTK file for saving solution
vtkfile = File('darcy1/add_mass.pvd')

#Time_stepping
M = Function(V)
t = dt
for n in range(num_steps):
    print(t)
    # Update current time
    t += dt
    S.t = t
    M_D.t = t
    # Compute solution
    solve(a==L, M, bc)
    #Save to file and plot solution
    vtkfile  << (M,t)
    # Update previous solution
    M_n.assign(M)

print(np.sqrt(assemble((M_n-M_D)*(M_n-M_D)*dx)))