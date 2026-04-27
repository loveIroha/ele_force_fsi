from mglib import restrict,restrict_bc,interpolate,compute_multigrid_levels_3D
from poisson import residual as compute_residual
from poisson import smooth
import numpy as np

class MG():
    def __init__(self, Nx, Ny, Nz, width, height, depth, boundary_type,
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
        num_levels, level_dh, level_dim = compute_multigrid_levels_3D(
            Nx, Ny,Nz, width, height, depth)
        # print(f"num_levels: {num_levels}")
        # print(f"level_dh: {level_dh}")
        # print(f"level_dim: {level_dim}")

        # 分配每一层网格需使用的变量
        multi_x = []
        multi_e = []
        multi_r = []
        multi_b = []
        for dim in level_dim:
            multi_x.append(np.zeros((dim[0]+2, dim[1]+2, dim[2]+2)))
            multi_e.append(np.zeros((dim[0]+2, dim[1]+2, dim[2]+2)))
            multi_r.append(np.zeros((dim[0]+2, dim[1]+2, dim[2]+2)))
            multi_b.append(np.zeros((dim[0]+2, dim[1]+2, dim[2]+2)))

        # 计算每一层网格的边界类型
        multi_bc = []
        multi_bc.append(boundary_type)
        for i in range(1, num_levels):
            dim = level_dim[i]
            print(f"level {dim}")
            multi_bc.append(restrict_bc(multi_bc[i-1], dim[0], dim[1], dim[2]))

        # 为类的成员变量赋值
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.width = width
        self.height = height
        self.depth = depth
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

    def test_restrict(self):
        for i in range(1, self.num_levels):
            dim = self.level_dim[i]
            print(f"level {dim}")
            self.multi_r[i] = restrict(self.multi_r[i-1],dim[0],dim[1],dim[2])
        
        for i in range(1, self.num_levels):
            dim = self.level_dim[i]
            print(f"level {dim}")
            self.multi_r[i-1] = interpolate(self.multi_r[i],dim[0],dim[1],dim[2])
            
    def two_grid_iteration(self, x, b):
        current_level = 0
        width, height, depth = self.width, self.height, self.depth
        Nx, Ny, Nz = self.level_dim[current_level][0], self.level_dim[current_level][1],self.level_dim[current_level][2]
        Nx_coarse, Ny_coarse, Nz_coarse = self.level_dim[current_level+1][0], self.level_dim[current_level+1][1],self.level_dim[current_level+1][2]
        self.multi_x[0] = x.copy()
        self.multi_b[0] = b.copy()
        self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_b[current_level], self.multi_bc[current_level], Nx, Ny, Nz, width, height, depth)
        self.multi_r[current_level] = compute_residual(self.multi_x[current_level],self.multi_b[current_level], self.multi_bc[current_level],  Nx, Ny, Nz, width, height,depth)
        self.multi_b[current_level + 1] = restrict(self.multi_r[current_level], Nx_coarse, Ny_coarse, Nz_coarse)
        self.multi_x[current_level + 1] = smooth(self.multi_x[current_level + 1], self.multi_b[current_level + 1],self.multi_bc[current_level + 1], Nx_coarse, Ny_coarse, Nz_coarse, width, height, depth)
        self.multi_e[current_level] = interpolate(self.multi_x[current_level + 1], Nx_coarse, Ny_coarse, Nz_coarse)
        self.multi_x[current_level] = self.multi_x[current_level] + self.multi_e[current_level]
        self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_b[current_level], self.multi_bc[current_level], Nx, Ny, Nz, width, height, depth)
        return self.multi_r[current_level]
    
    def iterate(self, x, b, current_level):
        
        width, height, depth = self.width, self.height, self.depth
        Nx, Ny, Nz = self.level_dim[current_level][0], self.level_dim[current_level][1],self.level_dim[current_level][2]
        
        if len(x) != len(b):
            raise ValueError("WRONG size.")

        if current_level == 0:
            self.multi_x[0] = x.copy()
            self.multi_b[0] = b.copy()

        for _ in range(self.num_presmooth):
            self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_b[current_level], self.multi_bc[current_level], 
                                                 Nx, Ny, Nz, width, height, depth)
            
# def smooth(p,bp, boundary_type, Nx, Ny, Nz, width, height, depth):
            
        if current_level < self.num_levels - 1:
            Nx_coarse, Ny_coarse, Nz_coarse = self.level_dim[current_level+1][0], self.level_dim[current_level + 1][1], self.level_dim[current_level + 1][2]
            
            self.multi_x[current_level + 1][:] = 0.0

            self.multi_r[current_level] = compute_residual(
                self.multi_x[current_level],self.multi_b[current_level], self.multi_bc[current_level],  Nx, Ny, Nz, width, height,depth)

            self.multi_b[current_level + 1] = restrict(self.multi_r[current_level],
                                                       Nx_coarse, Ny_coarse, Nz_coarse)

            self.multi_x[current_level + 1] = self.iterate(
                self.multi_x[current_level + 1], self.multi_b[current_level + 1], current_level + 1)

            self.multi_e[current_level] = interpolate(
                self.multi_x[current_level + 1], Nx_coarse, Ny_coarse, Nz_coarse)

            self.multi_x[current_level] = self.multi_x[current_level] + \
                self.multi_e[current_level]
        else:
            for _ in range(self.num_exactsmooth):
                self.multi_x[current_level] = smooth(self.multi_x[current_level],self.multi_b[current_level], self.multi_bc[current_level], 
                                                     Nx, Ny, Nz, width, height, depth)

        for _ in range(self.num_aftersmooth):
            self.multi_x[current_level] = smooth(self.multi_x[current_level], self.multi_b[current_level], self.multi_bc[current_level],
                                                 Nx, Ny, Nz, width, height, depth)

        return self.multi_x[current_level]