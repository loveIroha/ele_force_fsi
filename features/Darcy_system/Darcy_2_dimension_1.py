"""求解方程组
      div W = S
      W = -J*F_1*K*F_T*grad(P)
    给定变形梯度F=[[1, 2], [3, 4]],验证方程的收敛性"""



from fenics import *
from mshr import *
import ufl
import numpy as np

n = 64
mesh = UnitSquareMesh(n,n)
V = FunctionSpace(mesh, 'P',1)
# VT = TensorFunctionSpace(mesh, 'P', 1)
VT = VectorFunctionSpace(mesh, 'P', 1)

S = Constant(40.0)
K = 1.0*Identity(2)

#Define boundary condition
p_D = Expression('1+x[0]*x[0] + 2*x[1]*x[1]', degree=2)
w_D = Expression(('20*x[0]-28*x[1]','-14*x[0]+20*x[1]'), degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, p_D, boundary)

#Define variational problem
p = TrialFunction(V)
v = TestFunction(V)

F = as_matrix([[1, 2], [3, 4]])
J = ufl.det(F)
F_1 = ufl.inv(F)
F_T = ufl.transpose(F_1)

F = inner((J*F_1*K*F_T*grad(p)),grad(v))*dx - S*v*dx
# F = inner(F*         grad(p) ,grad(v))*dx - S*v*dx
a = lhs(F)
L = rhs(F)

p = Function(V)
solve(a==L, p, bc)
print(np.sqrt(assemble((p-p_D)*(p-p_D)*dx)))
W = -J*F_1*K*F_T*grad(p)
W = project(W, VT)
print(np.sqrt(assemble(inner(W-w_D,W-w_D)*dx)))
#Save solution to file in VTK format
vtkfile = File('darcy/pressure.pvd')
vtkfile << p
vtkfile = File('darcy/velocity.pvd')
vtkfile << W

# File('darcy/pressure_EXACT.pvd') << project(p_D,V)
# import numpy as np
# matrix = np.array([[1,2],[3,4]])
# F_1 = np.linalg.inv(matrix)
# F_T = np.transpose(F_1)
# print(F_1)
# print(F_T)
# print(-2*np.dot(F_1,F_T))