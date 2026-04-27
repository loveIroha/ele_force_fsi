""""""


from fenics import *
from mshr import *
import ufl
import numpy as np

T = 2
num_steps = 10
dt = T / num_steps
M_b = 2.0
b = 1.0
rho = 1.0
kappa = 1.0
phi_0 = 1.0

n = 512
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'P', 1)
VT = VectorFunctionSpace(mesh, 'P', 1)

S = Expression('2.9629629629629628*pow(x[0], 2)/pow(t + 0.33333333333333331*pow(x[0], 2) + 0.66666666666666663*pow(x[1], 2) + 0.66666666666666663, 3) - 8.2962962962962941*x[0]*x[1]/pow(t + 0.33333333333333331*pow(x[0], 2) + 0.66666666666666663*pow(x[1], 2) + 0.66666666666666663, 3) + 5.9259259259259256*pow(x[1], 2)/pow(t + 0.33333333333333331*pow(x[0], 2) + 0.66666666666666663*pow(x[1], 2) + 0.66666666666666663, 3) - 46.096451110408751 - 4.4444444444444446/pow(t + 0.33333333333333331*pow(x[0], 2) + 0.66666666666666663*pow(x[1], 2) + 0.66666666666666663, 2)', degree=5, t=0)
K = 1.0*Identity(2)

#Define boundary condition
M_D = Expression('1.0+x[0]*x[0] + 2.0*x[1]*x[1] + 3.0*t', degree=2, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, M_D, boundary)

# jit just in time compiling

#Define variational problem
M = TrialFunction(V)
N = TestFunction(V)
M_n = interpolate(M_D, V)

F = as_matrix([[-1, -2], [3, 4]])
J = ufl.det(F)
F_1 = ufl.inv(F)
F_T = ufl.transpose(F_1)

def f(J):
    return 2*(J-1-ufl.ln(J))/(J-1)*(J-1)

def p(M,M_n,J):
    return M_b*(b*(1-J) + M/rho)*f(J)-kappa*(rho/(M_n+rho*phi_0))

F = (M-M_n)/dt*N*dx+inner((J*F_1*K*F_T*grad(p(M,M_n,J))),grad(N))*dx - S*N*dx
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
    
    

