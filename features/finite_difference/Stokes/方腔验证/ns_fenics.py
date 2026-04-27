from dolfin import *

# Discretization parameters

def fun(T, dt, N, x, y):
    # Create mesh
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    nu = 0.01

    # Define boundaries
    upflow = 'near(x[1], 1.0, DOLFIN_EPS)'
    wall = 'near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0)'

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Constant((1, 0)), upflow)
    bcu_wall = DirichletBC(V, Constant((0, 0)), wall)

    bcu = [bcu_inflow, bcu_wall]
    bcp = []

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0))

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
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

    # Time-stepping
    t = 0
    while t < T - DOLFIN_EPS:
        t += dt
        print(t)

        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "bicgstab", "amg")

        # Pressure correction
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(A2, p1.vector(), b2, "bicgstab", "amg")
        p1.vector()[:] -= assemble(p1*dx)/assemble(1*dx(domain=mesh))

        # Velocity correction
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "bicgstab", "amg")
        # Move to next time step
        u0.assign(u1)

    for j in range(34):
        u0.set_allow_extrapolation(True)
        print(u0((x, (j-0.5)/32))[0])

T = 0.1
fun(T, 0.005, 32, 0.5, 0.5)

