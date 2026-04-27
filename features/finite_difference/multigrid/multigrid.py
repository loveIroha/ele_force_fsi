import numpy as np
from localtools import NEUMANN, DIRICHLET


from mglib import *


class MG():
    def __init__(self, Nx, Ny, width, height, boundary_type,
                 grid_type="cell-centered",
                 tol_relative=1e-6,
                 tol_descending=1e-6,
                 num_presmooth=2,
                 num_aftersmooth=2,
                 num_exactsmooth=100,
                 max_iteration=50,
                 N_coarsest=3
                  ):

        # 计算网格层数和每一层网格的dh和dim
        num_levels, level_dh, level_dim = compute_multigrid_levels_2D(
            Nx, Ny, width, height)

        # 分配每一层网格需使用的变量
        multi_x = []
        multi_e = []
        multi_r = []
        multi_b = []
        for dim in level_dim:
            multi_x.append(np.zeros((dim[0]+2, dim[1]+2)))
            multi_e.append(np.zeros((dim[0]+2, dim[1]+2)))
            multi_r.append(np.zeros((dim[0]+2, dim[1]+2)))
            multi_b.append(np.zeros((dim[0]+2, dim[1]+2)))

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
                                                 self.level_dim[current_level][0], self.level_dim[current_level][1], width, height)
        if current_level < self.num_levels - 1:
            self.multi_x[current_level + 1][:] = 0.0

            self.multi_r[current_level] = compute_residual(
                self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level], self.level_dim[current_level][0], self.level_dim[current_level][1], width, height)

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
                                                     self.level_dim[current_level][0], self.level_dim[current_level][1], width, height)

        for _ in range(self.num_aftersmooth):
            self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_bc[current_level], self.multi_b[current_level],
                                                 self.level_dim[current_level][0], self.level_dim[current_level][1], width, height)

        return self.multi_x[current_level]


def get_boundary_type(Nx, Ny, all_boundary_type):
    boundary_type = np.zeros((Nx+2, Ny+2))

    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary_type[0]
        boundary_type[i][Ny+1] = all_boundary_type[1]

    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary_type[2]
        boundary_type[Nx+1][j] = all_boundary_type[3]

    return boundary_type


if __name__ == "__main__":
    Nx = 256
    Ny = 256
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    all_boundary_type = [ DIRICHLET, NEUMANN,  NEUMANN, DIRICHLET]
    boundary_type = get_boundary_type(Nx, Ny, all_boundary_type)

    mg_solver = MG(Nx, Ny, width, height, boundary_type)

    from localtools import u_1
    from pressure import make_cell_centered_demo
    phi_prime, b, u_exact, boundary_type = make_cell_centered_demo(
        Nx, Ny, width, height, u_1, all_boundary_type)
    
    np.savetxt('boundary_type.txt', boundary_type)
    np.savetxt('u_exact.txt', u_exact)
    np.savetxt('b.txt', b)
    np.savetxt('phi_prime.txt', phi_prime)

    x = np.zeros((Nx+2, Ny+2))
    error = np.zeros((Nx+2, Ny+2))
    for _ in range(10):
        x = mg_solver.iterate(x, b, 0)
        phi = x + phi_prime
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                error[i][j] = phi[i][j] - u_exact[i][j]

        error_norm_l2 = np.sqrt(np.sum(error**2)*dx*dy)

        print(f"error norm l2: {error_norm_l2}")
    
    np.savetxt('python_result.txt', phi)
    
