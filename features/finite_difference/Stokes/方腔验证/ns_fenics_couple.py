from fenics import *
import numpy as np
import time

def fun(T, dt, N, x, y):
    # Create mesh
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)

    # Define function spaces
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    # Set parameter values
    nu = 0.01

    # Define boundaries
    upflow = 'near(x[1], 1.0, DOLFIN_EPS)'
    wall = 'near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0)'

    # Define boundary conditions
    bcu_inflow = DirichletBC(W.sub(0), Constant((1, 0)), upflow)
    bcu_wall = DirichletBC(W.sub(0), Constant((0, 0)), wall)

    bcu = [bcu_inflow, bcu_wall]
    bcp = []



    k = Constant(dt)
    f = Constant((0, 0))
    wn = Function(W)
    (un, _) = wn.split(True)    # from now on, wn can't be used.

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    F = inner((u-un)/k, v)*dx + inner(grad(un)*un, v)*dx + nu * \
        inner(grad(u), grad(v))*dx - div(v)*p * \
        dx + q*div(u)*dx - inner(f, v)*dx
    a = lhs(F)
    L = rhs(F)

    # Assemble matrix and vector
    A = assemble(a)

    t = 0
    w_ = Function(W)
    while t < T - DOLFIN_EPS:
        t += dt
        print(t)
        # Assemble right side term
        b = assemble(L)
        # Compute solution
        [bc.apply(A, b) for bc in bcu]
        solve(A, w_.vector(), b)
        # Update coefficients
        (u_, p_) = w_.split(True)
        un.assign(u_)


    for j in range(34):
        un.set_allow_extrapolation(True)
        print(un((x, (j-0.5)/32))[0])

T = 0.1
fun(T, 0.005, 80, 0.5, 0.5)
