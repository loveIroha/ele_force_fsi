from localtools import u_1
from pressure import make_cell_centered_demo
import numpy as np


def restrict(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            coarse[i, j] = 0.25*(
                fine[2*i-1, 2*j-1]+fine[2*i, 2*j-1]+fine[2*i-1, 2*j]+fine[2*i, 2*j])
    return coarse


def compute_residual(phi, b, Nx=32, Ny=32, width=1.0, height=1.0):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
            p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
            residual[i][j] = b[i][j] - p1 - p2
    return residual


def interpolate(coarse, Nx, Ny):
    """interpolate v to the fine grid"""
    def L0(x): return 1.0 - x
    def L1(x): return x
    L = [L0, L1]
    fine = np.zeros((2*Nx+2, 2*Ny+2))

    def bilinear(x, y, f):
        sum = 0.0
        for i in range(0, 2):
            for j in range(0, 2):
                sum += f[j*2+i]*L[i](x)*L[j](y)
        return sum
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            f = [coarse[i, j], coarse[i+1, j],
                 coarse[i, j+1], coarse[i+1, j+1]]
            fine[2*i, 2*j] = bilinear(1.0/3.0, 1.0/3.0, f)
            fine[2*i, 2*j+1] = bilinear(1.0/3.0, 2.0/3.0, f)
            fine[2*i+1, 2*j] = bilinear(2.0/3.0, 1.0/3.0, f)
            fine[2*i+1, 2*j+1] = bilinear(2.0/3.0, 2.0/3.0, f)
    return fine


def compute_multigrid_levels_3D(Nx=18, Ny=16, Nz=10, width=1.0, height=1.0, depth=1.0):
    dh = (width/Nx, height/Ny, depth/Nz)
    num_levels = 0
    level_dh = []
    level_dim = []
    while True:
        num_levels += 1
        level_dh.append(dh)
        level_dim.append((Nx, Ny, Nz))
        print(
            f"num_levels : {num_levels}, level_dh  : {dh[0]:.4e}, {dh[1]:.4e}, {dh[2]:.4e}")
        print(f"num_levels : {num_levels}, level_dim : {Nx}, {Ny}, {Nz}")
        if Nx % 2 == 1 or Ny % 2 == 1 or Nz % 2 == 1:
            break
        if Nx * Ny * Nz < 65:
            break
        Nx, Ny, Nz = (Nx // 2, max(Ny // 2, 1), max(Nz // 2, 1))
        dh = (dh[0] * 2.0, dh[1] * 2.0, dh[2] * 2.0)
    if num_levels <= 1:
        raise Exception("the problem is too small for the multigrid solver.")


def compute_multigrid_levels_2D(Nx=18, Ny=16, width=1.0, height=1.0):
    num_levels = 0
    level_dh = []
    level_dim = []
    dx = width/Nx
    dy = height/Ny
    while True:
        num_levels += 1
        level_dh.append((dx, dy))
        level_dim.append((Nx, Ny))
        print(f"num_levels : {num_levels}, level_dh  : {dx:.4e}, {dy:.4e}")
        print(f"num_levels : {num_levels}, level_dim : {Nx}, {Ny}")
        if Nx % 2 == 1 or Ny % 2 == 1:
            break
        if Nx * Ny < 17:
            break
        Nx, Ny = (Nx // 2, Ny // 2)
        dx, dy = (dx * 2, dy * 2)
    if num_levels <= 1:
        raise Exception("the problem is too small for the multigrid solver.")
    return num_levels, level_dh, level_dim


def init_multigrid_solver(level_dh, level_dim):
    multi_x = []
    multi_e = []
    multi_r = []
    multi_b = []
    multi_bc = []
    for dh, dim in zip(level_dh, level_dim):
        multi_x.append(np.zeros((dim[0]+2, dim[1]+2)))
        multi_e.append(np.zeros((dim[0]+2, dim[1]+2)))
        multi_r.append(np.zeros((dim[0]+2, dim[1]+2)))
        multi_b.append(np.zeros((dim[0]+2, dim[1]+2)))
        multi_bc.append(np.zeros((dim[0]+2, dim[1]+2)))
    return multi_x, multi_e, multi_r, multi_b, multi_bc


def restrict_bc(fine, Nx, Ny):
    """Restrict boundary type to the coarse grid"""
    coarse = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        coarse[i][0] = max(fine[2*i-1, 0], fine[2*i, 0])
        coarse[i][Ny+1] = max(fine[2*i-1, 2*Ny+1], fine[2*i, 2*Ny+1])
    for j in range(1, Ny+1):
        coarse[0][j] = max(fine[0, 2*j-1], fine[0, 2*j])
        coarse[Nx+1][j] = max(fine[2*Nx+1, 2*j-1], fine[2*Nx+1, 2*j])
    return coarse


def smooth(phi, boundary_type, b, Nx, Ny, width, height):
    dx = width/Nx
    dy = height/Ny
    for i in range(1, Nx+1):
        phi[i][0] = (boundary_type[i][0] == DIRICHLET)*(-phi[i][1]
                                                        ) + (boundary_type[i][0] == NEUMANN)*phi[i][1]
        phi[i][Ny+1] = (boundary_type[i][Ny+1] == DIRICHLET) * \
            (-phi[i][Ny]) + (boundary_type[i][Ny+1] == NEUMANN)*phi[i][Ny]

    for j in range(1, Ny+1):
        phi[0][j] = (boundary_type[0][j] == DIRICHLET) * \
            (-phi[1][j])+(boundary_type[0][j] == NEUMANN)*phi[1][j]
        phi[Nx+1][j] = (boundary_type[Nx+1][j] == DIRICHLET) * \
            (-phi[Nx][j])+(boundary_type[Nx+1][j] == NEUMANN)*phi[Nx][j]

    # 迭代求解
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            phi[i][j] = ((phi[i+1][j]+phi[i-1][j])*dy*dy+(phi[i][j+1] +
                                                          phi[i][j-1])*dx*dx-b[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)

    return phi


def iterate(x, b, current_level):
    if len(x) != len(b):
        raise ValueError("WRONG size.")

    if current_level == 0:
        multi_x[0] = x.copy()
        multi_b[0] = b.copy()

    # print(multi_bc[current_level])
    # print(multi_b[current_level])
    # print(level_dim[current_level])
    # print(level_dh[current_level])
    # for i in range(1000):
    #     multi_x[current_level] = smooth(multi_x[current_level], multi_bc[current_level], multi_b[current_level], level_dim[current_level][0], level_dim[current_level][1], width, height)

    for _ in range(num_presmooth):
        multi_x[current_level] = smooth(multi_x[current_level], multi_bc[current_level], multi_b[current_level],
                                        level_dim[current_level][0], level_dim[current_level][1], width, height)
    if current_level < num_levels - 1:
        multi_x[current_level + 1][:] = 0.0

        multi_r[current_level] = compute_residual(
            multi_x[current_level], multi_b[current_level], level_dim[current_level][0], level_dim[current_level][1], width, height)

        multi_b[current_level + 1] = restrict(multi_r[current_level],
                                              level_dim[current_level+1][0], level_dim[current_level + 1][1])

        multi_x[current_level + 1] = iterate(
            multi_x[current_level + 1], multi_b[current_level + 1], current_level + 1)

        multi_e[current_level] = interpolate(
            multi_x[current_level + 1], level_dim[current_level+1][0], level_dim[current_level + 1][1])

        multi_x[current_level] = multi_x[current_level] + \
            multi_e[current_level]
    else:
        for _ in range(num_exactsmooth):
            multi_x[current_level] = smooth(multi_x[current_level], multi_bc[current_level], multi_b[current_level],
                                            level_dim[current_level][0], level_dim[current_level][1], width, height)

    for _ in range(num_aftersmooth):
        multi_x[current_level] = smooth(multi_x[current_level], multi_bc[current_level], multi_b[current_level],
                                        level_dim[current_level][0], level_dim[current_level][1], width, height)

    return multi_x[current_level]

tol_relative=1e-6
tol_descending=1e-6
num_presmooth=2
num_aftersmooth=2
num_exactsmooth=100
max_iteration=50
# class MG():
#     N_coarsest = 3

#     def __init__(self, Nx, Ny, width, height, boundary_type,
#                  grid_type="cell-centered",
#                  tol_relative=1e-6,
#                  tol_descending=1e-6,
#                  num_presmooth=2,
#                  num_aftersmooth=2,
#                  num_exactsmooth=100,
#                  max_iteration=50
#                  ):
#         self.Nx = Nx
#         self.Ny = Ny
#         self.width = width
#         self.height = height
#         self.boundary_type = boundary_type
#         self.grid_type = grid_type
#         self.tol_relative = tol_relative
#         self.tol_descending = tol_descending
#         self.num_presmooth = num_presmooth
#         self.num_aftersmooth = num_aftersmooth
#         self.num_exactsmooth = num_exactsmooth
#         self.max_iteration = max_iteration



#         pass


Nx = 512
Ny = 512
width = 1.0
height = 1.0


NEUMANN = 2
DIRICHLET = 1


all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET]
boundary_type = np.zeros((Nx+2, Ny+2))

for i in range(1, Nx+1):
    boundary_type[i][0] = all_boundary_type[0]
    boundary_type[i][Ny+1] = all_boundary_type[1]

for j in range(1, Ny+1):
    boundary_type[0][j] = all_boundary_type[2]
    boundary_type[Nx+1][j] = all_boundary_type[3]

num_levels, level_dh, level_dim = compute_multigrid_levels_2D(Nx, Ny)
multi_x, multi_e, multi_r, multi_b, multi_bc = init_multigrid_solver(
    level_dh, level_dim)
    
multi_bc = []
multi_bc.append(boundary_type)
for i in range(1, num_levels):
    dim = level_dim[i]
    print(f"level {dim}")
    multi_bc.append(restrict_bc(multi_bc[i-1], dim[0], dim[1]))


phi_prime, b, u_exact, boundary_type = make_cell_centered_demo(
    Nx, Ny, width, height, u_1, all_boundary_type)

dx = width/Nx
dy = height/Ny

x = np.zeros((Nx+2, Ny+2))
error = np.zeros((Nx+2, Ny+2))
# print(boundary_type)
# for i in range(10000):
#     x = smooth(x, boundary_type, b, Nx, Ny, width, height)

x = iterate(x, b, 0)
x = iterate(x, b, 0)
x = iterate(x, b, 0)
x = iterate(x, b, 0)
x = iterate(x, b, 0)
x = iterate(x, b, 0)
x = iterate(x, b, 0)
# x = iterate(x, b, 0)
phi = x + phi_prime


for i in range(1, Nx+1):
    for j in range(1, Ny+1):
        error[i][j] = phi[i][j] - u_exact[i][j]

error_norm_l2 = np.sqrt(np.sum(error**2)*dx*dy)

print(f"error norm l2: {error_norm_l2}")
