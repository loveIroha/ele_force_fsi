from fenics import *
from mshr import *

domain = Box(Point(0.95, 0.95, 0.5), Point(1.05, 1.05, 1.5))
mesh = generate_mesh(domain,10)
# mesh = BoxMesh(Point(0,0,0),Point(1,1,1),8,8,8)
n = FacetNormal(mesh)

# Interpolate mesh values into boundary function
# u_boundary = Function(D)
# u_boundary.set_allow_extrapolation(True)

# Approximate facet normal in a suitable space using projection
n = FacetNormal(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
u_ = TrialFunction(V)
v_ = TestFunction(V)
a = inner(u_,v_)*dx
l = inner(n, v_)*ds
A = assemble(a, keep_diagonal=True)
L = assemble(l)

A.ident_zeros()
nh = Function(V)

solve(A, nh.vector(), L, "gmres", "amg")

File("outnormal.pvd") << nh

# class normal_u(UserExpression):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#     def eval(self, values, x):
#         n_eval = nh(x)
#         u_eval = u(x)
#         un = (u_eval[0]*n_eval[0] + u_eval[1]*n_eval[1])
#         values[0] = un * n_eval[0]
#         values[1] = un * n_eval[1]

#     def value_shape(self):
#         return (2,)

# u_boundary.interpolate(normal_u())