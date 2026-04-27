from dolfin import *


def fun(T, dt, N, x, y):
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Set parameter values
    nu = 1.0

    # Define boundaries
    upflow = 'near(x[1], 1.0, DOLFIN_EPS)'
    wall = 'near(x[0], 0.0) || near(x[0], 1.0) || near(x[1], 0.0)'

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Constant((1, 0)), upflow)
    bcu_wall = DirichletBC(V, Constant((0, 0)), wall)
    bcu = [bcu_inflow, bcu_wall]

    # Create functions
    u0 = Function(V)
    u_ = Function(V)

    # Define coefficients
    k = Constant(dt)

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + nu*inner(grad(u), grad(v))*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    A1 = assemble(a1)
    t = 0
    while t < T - DOLFIN_EPS:
        t += dt
        print(t)
        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u_.vector(), b1)

        u0.assign(u_)

    for j in range(34):
        u0.set_allow_extrapolation(True)
        print(u0((x, (j-0.5)/32))[0])
    return u0((x, y))[0]


T = 0.1
fun(T, 0.005, 10, 0.5, 0.5)

# if __name__ == "__main__":
#     list_value = []
#     list_dt = []
#     for i in range(3,6):
#         values = []
#         dts = []
#         for j in range(4,7):
#             x, y = 0.5, 0.484375
#             T = 0.1
#             N = 10+1 << i
#             dt = T/(1 << j)
#             print("dt : ", dt, "N : ", N)
#             X = fun(T, dt, N, x, y)
#             values.append(X)
#             dts.append(dt)
#             print("dt : ", dt, "N : ", N, "Value : ", X)

#         list_dt.append(dts)
#         list_value.append(values)

#     from localtools import plot_convergence
#     plot_convergence(list_dt,list_value)