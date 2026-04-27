from fenics import *

mesh = UnitSquareMesh(1, 1)

V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v) * dx
L = Constant(1) * v * dx

A = assemble(a)
b = assemble(L)

c = Vector(b)
d = A*b

# the following code will demonstrate that the result of A*c is euqal to b