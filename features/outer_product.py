from fenics import *

mesh = UnitCubeMesh(2,2,2)

V = VectorFunctionSpace(mesh, "Lagrange", 1)
T = TensorFunctionSpace(mesh, "Lagrange", 1)

f = Expression(("1.0", "2.0","3.0"),degree=1)
g = Expression(("3.0", "2.0","1.0"),degree=1)

fog = outer(f, g)
f = project(fog, T)

File("f.pvd") << f
