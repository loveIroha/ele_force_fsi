
from dolfin import *

# Load mesh from file
mesh = BoxMesh(Point(0,0,0),Point(1,1,1),40,40,40)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
Nt = 64
T = 1
dt = T/Nt
nu = 0.001

#Define boundaries
upflow   = 'near(x[1], 1.0)'
wall     = 'near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0) || near(x[2], 0.0) || near(x[2], 1.0)'

#Define boundary conditions
bcu_inflow = DirichletBC(V, Constant((1,0,0)), upflow)
bcu_wall = DirichletBC(V, Constant((0,0,0)), wall)

bcu = [bcu_inflow, bcu_wall]
bcp = []

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(dot(u0,grad(u0)),v)*dx + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
F1 = (1/k)*inner(u - u0, v)*dx                                + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:
    print(t)

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")

    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)

    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")

    # Move to next time step
    u0.assign(u1)
    t += dt


u0.set_allow_extrapolation(True)
p1.set_allow_extrapolation(True)
N = 32
dx = 1.0/N
for i in range(N):
    print(u0(((N//2)*dx,i*dx-0.5*dx,N//2*dx-0.5*dx))[0])


# print('u max:',u0.vector().max(),'u center:', u0((0.5,0.5,0.5))[0])
# print('p max:',p1.vector().max(),'p center:', p1((0.5,0.5,0.5)))