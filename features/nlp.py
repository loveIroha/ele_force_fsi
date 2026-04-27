from fenics import *

mesh=UnitSquareMesh(5,5)
V = FunctionSpace(mesh,"P",1)
v = TestFunction(V)
u = Function(V)

def q(u):
    return 1+u*u

u.interpolate(Expression("x[0]", degree=1))
f = Expression('0', degree=0)

H = q(u)*dot(grad(u), grad(v))*dx -f*v*dx
L = dot(grad(u), grad(v))*dx -f*v*dx

b = assemble(F)


