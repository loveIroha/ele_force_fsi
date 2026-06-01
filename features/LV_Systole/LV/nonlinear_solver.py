
"""Solve the Yamabe PDE which arises in the differential geometry of
general relativity. http://arxiv.org/abs/1107.0360.

The Yamabe equation is highly nonlinear and supports many
solutions. However, only one of these is of physical relevance -- the
positive solution.

This unit test demonstrates the capability of the SNES solver to
accept bounds on the resulting solution. The plain Newton method
converges to an unphysical negative solution, while the SNES solution
with {sign: nonnegative} converges to the physical positive solution.

An alternative interface to SNESVI allows the user to set explicitly
more complex bounds as GenericVectors or Function.
"""

from dolfin import *
parameter_backend = []
def test_preconditioner_interface(V, parameter_backend):
    "Test nonlinear solvers preconditioner interface"
    class Problem(NonlinearProblem):
        def __init__(self, V):
            NonlinearProblem.__init__(self)

            u = Function(V)
            u_ = TrialFunction(V)
            v = TestFunction(V)
            L = Constant(1.0)*v*dx
            bc = DirichletBC(V, 0, "on_boundary")

            # Nonlinear problem and its Jacobian
            F = (1.0+u*u)*inner(grad(u), grad(v))*dx - L
            # J = derivative(F, u)
            J = derivative(F, u, u_)

            assembler = SystemAssembler(J, F, bc)

            self.u = u
            self.assembler = assembler

        def F(self, b, x):
            self.assembler.assemble(b, x)

        def J(self, A, x):
            self.assembler.assemble(A)

    for solverclass in [NewtonSolver]:
        problem = Problem(V)
        x = problem.u.vector()

        solver = solverclass()
        solver.parameters["linear_solver"] = "bicgstab"
        solver.parameters["preconditioner"] = "amg"
        solver.parameters["krylov_solver"]["monitor_convergence"] = True
        solver.parameters["report"] = True

        # Check that subsequent solutions work and reuse preconditioner
        solver.solve(problem, x)
        # 逐渐增加的大小
        # x.zero()
        nliter, nlconv = solver.solve(problem, x)
        print("nliter: %d, nlconv: %d" % (nliter, nlconv))

        # Check that overloading NewtonSolver members works
        getattr(solver, "check_overloads_called", None)


mesh = UnitCubeMesh(20, 20, 20)
V = FunctionSpace(mesh, "CG", 1)
test_preconditioner_interface(V, parameter_backend) # Test preconditioner interface




# from fenics import *
# from ufl import *


# class Problem(NonlinearProblem):
#     def __init__(self, V):
#         # 王永恒硕士毕业论文中使用的参数
#         a = 2400
#         b = 5.08
#         a_f = 14600
#         b_f = 4.15
#         a_s = 8700
#         b_s = 1.6
#         a_fs = 3000
#         b_fs = 1.3
#         beta_s = 5e6 

#         V = VectorElement("Lagrange", tetrahedron, 1)
#         S = FiniteElement("Lagrange", tetrahedron, 1)
#         R = FiniteElement("Real",     tetrahedron, 0)

#         f0 = Coefficient(V)                 # fiber direction
#         s0 = Coefficient(V)                 # sheet direction
#         X  = Coefficient(V)                 # current position, the variable to be solved
#         x_start  = Coefficient(V)           # initial position

#         F = variable(grad(X)) 
#         C = F.T*F
#         J = det(F)
#         I1 = tr(C)
#         I3 = det(C)

#         I_4f = max_value(inner(f0, C*f0), 1.0)
#         I_4s = max_value(inner(s0, C*s0), 1.0)
#         I_8fs = inner(f0, C*s0)

#         W = a/2.0/b*exp(b*(I1-3))
#         W += a_f/2.0/b_f*(exp(b_f*(I_4f-1.0)*(I_4f-1.0))-1.0)
#         W += a_s/2.0/b_s*(exp(b_s*(I_4s-1.0)*(I_4s-1.0))-1.0)
#         W += a_fs/2.0/b_fs*(exp(b_fs*I_8fs*I_8fs)-1.0)

#         P = diff(W, F)

#         # incompressibility 
#         P -= a*exp(b*(I1-3))*inv(F).T

#         # incompressibility
#         P += beta_s*ln(I3)*inv(F).T

#         # Body force
#         v = TestFunction(V)
#         u = TrialFunction(V)
#         H = inner(P, grad(v))*dx(degree=5)

#         # Input the endocardium pressure load
#         pressure = Coefficient(R)

#         # Define the out normal at reference configuration.
#         N = FacetNormal(tetrahedron)

#         # Define the endocardium pressure
#         H = inner(pressure*det(F)*inv(F)*N,v)*ds(subdomain_id=2,degree=5)

#         # # Define the active contraction force
#         # T  = Coefficient(R)
#         # H += J*T*inner(outer(f0,f0), grad(v))*dx(degree=5)

#         # Define the boundary condition at the base
#         R = (x_start[0] - 7.30042735, x_start[1] - 7.6448494, 0)
#         r = (X[0]       - 7.30042735,       X[1] - 7.6448494, 0)
#         direction_length = (R[0]*r[0] + R[1]*r[1])/(R[0]*R[0]+R[1]*R[1])
#         x_constraint = direction_length*R[0]-r[0]
#         y_constraint = direction_length*R[1]-r[1]
#         z_constraint = (x_start[2] - X[2])
#         H -= 1e6*inner(as_vector((x_constraint, y_constraint, z_constraint)),v)*ds(subdomain_id=3,degree=5)

#         J = derivative(H, X, u)

#         self.H = H
#         self.J = J

#         assembler = SystemAssembler(J, H)

#         self.u = u
#         self.assembler = assembler

#         def F(self, b, x):
#             self.assembler.assemble(b, x)

#         def J(self, A, x):
#             self.assembler.assemble(A)