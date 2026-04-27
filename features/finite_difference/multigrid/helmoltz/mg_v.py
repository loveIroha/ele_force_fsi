import numpy as np
from localtools import NEUMANN, DIRICHLET
from mglib import compute_multigrid_levels_2D
from solve_demo import smooth_v as smooth
from mglib_v import restrict, interpolate, restrict_bc


def compute_residual(phi, boundary_type, vb,  Nx, Ny, width, height, dt, rho, mu):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+2, Ny+1))
    # 左右边界
    for j in range(0, Ny+1):
        # if boundary_type[0][j] == DIRICHLET:
        #     phi[0][j] = -phi[1][j]
        if boundary_type[0][j] == NEUMANN:
            phi[0][j] = phi[1][j]
        # if boundary_type[Nx+1][j] == DIRICHLET:
        #     phi[Nx+1][j] = -phi[Nx][j]
        if boundary_type[Nx+1][j] == NEUMANN:
            phi[Nx+1][j] = phi[Nx][j]

    # 上下边界
    for i in range(1, Nx+1):
        # if boundary_type[i][0] == DIRICHLET:
        #     phi[i][0] = 0.0
        if boundary_type[i][0] == NEUMANN:
            ghost_phi = phi[i][1]
            p1 = (phi[i-1][0]-2*phi[i][0]+phi[i+1][0])/dx/dx
            p2 = (ghost_phi-2*phi[i][0]+phi[i][1])/dy/dy
            residual[i][0] = vb[i][0] - phi[i][0]/dt + mu/rho*(p1+p2)
        # if boundary_type[i][Ny] == DIRICHLET:
        #     phi[i][Ny] = 0.0
        if boundary_type[i][Ny] == NEUMANN:
            ghost_phi = phi[i][Ny-1]
            p1 = (phi[i-1][Ny]-2*phi[i][Ny]+phi[i+1][Ny])/dx/dx
            p2 = (phi[i][Ny-1]-2*phi[i][Ny]+ghost_phi)/dy/dy
            residual[i][Ny] = vb[i][Ny] - phi[i][Ny]/dt + mu/rho*(p1+p2)

    for i in range(1, Nx+1):
        for j in range(1, Ny):
            p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
            p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
            residual[i][j] = vb[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)
    return residual

class MG():
    def __init__(self, Nx, Ny, width, height, dt, rho, mu, boundary_type,
                 grid_type="x-shifted",
                 tol_relative=1e-6,
                 tol_descending=1e-6,
                 num_presmooth=2,
                 num_aftersmooth=2,
                 num_exactsmooth=100,
                 max_iteration=50,
                 N_coarsest=3
                 ):
        self.dt = dt
        self.rho = rho
        self.mu = mu
        # 计算网格层数和每一层网格的dh和dim
        num_levels, level_dh, level_dim = compute_multigrid_levels_2D(
            Nx, Ny, width, height)

        # 分配每一层网格需使用的变量
        multi_x = []
        multi_e = []
        multi_r = []
        multi_b = []
        for dim in level_dim:
            multi_x.append(np.zeros((dim[0]+2, dim[1]+1)))
            multi_e.append(np.zeros((dim[0]+2, dim[1]+1)))
            multi_r.append(np.zeros((dim[0]+2, dim[1]+1)))
            multi_b.append(np.zeros((dim[0]+2, dim[1]+1)))

        # 计算每一层网格的边界类型
        multi_bc = []
        multi_bc.append(boundary_type)
        for i in range(1, num_levels):
            dim = level_dim[i]
            print(f"level {dim}")
            multi_bc.append(restrict_bc(multi_bc[i-1], dim[0], dim[1]))

        # 为类的成员变量赋值
        self.Nx = Nx
        self.Ny = Ny
        self.width = width
        self.height = height
        self.boundary_type = boundary_type
        self.grid_type = grid_type
        self.tol_relative = tol_relative
        self.tol_descending = tol_descending
        self.num_presmooth = num_presmooth
        self.num_aftersmooth = num_aftersmooth
        self.num_exactsmooth = num_exactsmooth
        self.max_iteration = max_iteration
        self.N_coarsest = N_coarsest
        self.num_levels = num_levels
        self.level_dh = level_dh
        self.level_dim = level_dim
        self.multi_x = multi_x
        self.multi_e = multi_e
        self.multi_r = multi_r
        self.multi_b = multi_b
        self.multi_bc = multi_bc

    def iterate(self, x, b, current_level):
        if len(x) != len(b):
            raise ValueError("WRONG size.")

        if current_level == 0:
            self.multi_x[0] = x.copy()
            self.multi_b[0] = b.copy()

        for _ in range(self.num_presmooth):
            self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level],
                                                   self.level_dim[current_level][0], self.level_dim[current_level][1],  self.width,  self.height, self.dt, self.rho, self.mu)
        if current_level < self.num_levels - 1:
            self.multi_x[current_level + 1][:] = 0.0

            self.multi_r[current_level] = compute_residual(
                self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level], self.level_dim[current_level][0], self.level_dim[current_level][1],  self.width,  self.height, self.dt, self.rho, self.mu)

            self.multi_b[current_level + 1] = restrict(self.multi_r[current_level],
                                                       self.level_dim[current_level+1][0], self.level_dim[current_level + 1][1])

            self.multi_x[current_level + 1] = self.iterate(
                self.multi_x[current_level + 1], self.multi_b[current_level + 1], current_level + 1)

            self.multi_e[current_level] = interpolate(
                self.multi_x[current_level + 1], self.level_dim[current_level+1][0], self.level_dim[current_level + 1][1])

            self.multi_x[current_level] = self.multi_x[current_level] + \
                self.multi_e[current_level]
        else:
            for _ in range(self.num_exactsmooth):
                self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level],
                                                       self.level_dim[current_level][0], self.level_dim[current_level][1],  self.width,  self.height, self.dt, self.rho, self.mu)

        for _ in range(self.num_aftersmooth):
            self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level],
                                                   self.level_dim[current_level][0], self.level_dim[current_level][1], self.width,  self.height, self.dt, self.rho, self.mu)

        return self.multi_x[current_level]


