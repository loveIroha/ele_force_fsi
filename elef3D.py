from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import max_value, min_value
# === 凹型网格生成 ===
def create_notched_mesh():
    mesh = BoxMesh(Point(0,0,0), Point(5,5,5), 5,5,5)
    class Notch(SubDomain):
        def inside(self, x, on_boundary):
            return (x[0] >= 1) & (x[0] <= 4) & (x[1] >= 1) & (x[1] <= 4) & (x[2] >= 1)
    markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    markers.set_all(0)
    Notch().mark(markers, 1)
    submesh = SubMesh(mesh, markers, 0)
    return submesh

mesh = create_notched_mesh()

# Function spaces
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
vecP1 = VectorElement('P', mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, MixedElement([P1, P1, vecP1]))
V = FunctionSpace(mesh, 'P', 1)
V_vec = VectorFunctionSpace(mesh, 'P', 1)

# GPB-LN parameters and state variables (simplified for demonstration)
params = {
    'C_m': 1.0,
    'G': 1.17e-3,
    'B_TnCL': 70.0,
    'eta': 0.5,
    'v_stim': 9.5
}

def deformation_gradient(U):
    I = Identity(3)
    return I + grad(U)

def compute_stress(U, active_stress):
    F = deformation_gradient(U)
    C = F.T * F
    J = det(F)
    S_passive = params['eta'] * (J**(-2/3)*C - Identity(3))
    fiber = Constant((1.0, 0.0, 0.0))
    lambda_f = sqrt(inner(fiber, C * fiber))
    S_active = active_stress / lambda_f * outer(fiber, fiber)
    return S_passive + S_active

# State vectors per cell: Vm, Ca, TRPN, active_stress
num_cells = mesh.num_cells()
y = np.zeros((num_cells, 4))

# Initial conditions
u = Function(ME)
u_old = Function(ME)
Vm, Ca_i, U = split(u)
Vm_old, Ca_i_old, U_old = split(u_old)

class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = 10.0 if near(x[0], 1.0) else -85.0
        values[1] = 0.107
        values[2] = 0.0
        values[3] = 0.0
        values[4] = 0.0
        # values[5] = 0.0
    def value_shape(self): return (5,)

u_init = InitialCondition(degree=2)
u.assign(interpolate(u_init, ME))
u_old.assign(u)

# Define test functions
w1, w2, w4 = TestFunctions(ME)

# I_ion (simple linear model)
def I_ion(Vm, Ca):
    return -0.05 * Vm + 0.001 * Ca

# Time stepping
dt = 0.1
t = 0.0
T_end = 650
step = 0

# Output
vm_pvd = File("3Dresults/Vm.pvd")
ca_pvd = File("3Dresults/Ca.pvd")
stress_pvd = File("3Dresults/Stress.pvd")
disp_pvd = File("3Dresults/Displacement.pvd")

# Boundary condition
bc_u = DirichletBC(ME.sub(2), Constant((0,0,0)), lambda x, on_boundary: on_boundary)

while t <= T_end:
    Vm_func, Ca_func, U_func = u.split()
    Vm_proj = project(Vm_func, V)
    Ca_proj = project(Ca_func, V)
    U_proj = project(U_func, V_vec)

    # # Update GPB-LN per cell ODE system
    # for cell in cells(mesh):
    #     cid = cell.index()
    #     Vm_val = Vm_proj.vector()[cid]
    #     Ca_val = Ca_proj.vector()[cid]
    #     # simplified active stress model
    #     y[cid, 0] = Vm_val
    #     y[cid, 1] = Ca_val
    #     y[cid, 2] += dt * (Ca_val / (0.52 + Ca_val) * (1 - y[cid, 2]) - y[cid, 2])  # TRPN ODE
    #     y[cid, 3] = params['B_TnCL'] * y[cid, 2]**3  # active stress
    for cell in cells(mesh):
        midpoint = cell.midpoint()
        Vm_val = Vm_proj(midpoint)
        Ca_val = Ca_proj(midpoint)
        cid = cell.index()
        y[cid, 0] = Vm_val
        y[cid, 1] = Ca_val
        y[cid, 2] += dt * (Ca_val / (0.52 + Ca_val) * (1 - y[cid, 2]) - y[cid, 2])  # TRPN ODE
        y[cid, 3] = params['B_TnCL'] * y[cid, 2]**3

    active_stress_func = Function(V)
    active_stress_func.vector().set_local(y[:, 3])

    S_total = compute_stress(U_func, active_stress_func)

    F_Vm = (params['C_m'] * (Vm - Vm_old)/dt * w1 * dx
            + inner(params['G'] * grad(Vm), grad(w1)) * dx
            - params['v_stim'] * w1 * dx
            + I_ion(Vm, Ca_i) * w1 * dx)

    F_Ca = (Ca_i - Ca_i_old)/dt * w2 * dx

    F_mech = inner(S_total, grad(w4)) * dx

    F = F_Vm + F_Ca + F_mech

    problem = NonlinearVariationalProblem(F, u, bcs=[bc_u], J=derivative(F, u))
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
    solver.solve()

    if step % 10 == 0:
        Vm_sol, Ca_sol, U_sol = u.split()
        Vm_proj = project(Vm_sol, V)
        Ca_proj = project(Ca_sol, V)
        U_proj = project(U_sol, V_vec)
        S_proj = project(compute_stress(U_proj, active_stress_func), TensorFunctionSpace(mesh, "P", 1))

        vm_pvd << (Vm_proj, t)
        ca_pvd << (Ca_proj, t)
        stress_pvd << (S_proj, t)
        disp_pvd << (U_proj, t)

        print(f"Time={t:.1f}ms, Max Vm={Vm_proj.vector().max():.2f}, Stress Norm={S_proj.vector().norm('l2'):.2e}")

    u_old.assign(u)
    t += dt
    step += 1

Vm_final = project(Vm_sol, V)
Stress_final = project(compute_stress(U_sol, active_stress_func), TensorFunctionSpace(mesh, "P", 1))
U_final = project(U_sol, V_vec)

plt.figure(figsize=(15,5))
plt.subplot(131)
plot(Vm_final, title="Membrane Potential")
plt.colorbar()
plt.subplot(132)
plot(Stress_final[0,0], title="Stress_xx Component")
plt.colorbar()
plt.subplot(133)
plot(U_final.sub(2), title="Displacement Z-component")
plt.colorbar()
plt.tight_layout()
plt.savefig("3Dresults/final_result.png", dpi=300, format='png')
print("后处理完成, 结果已保存至3Dresults/final_result.png")