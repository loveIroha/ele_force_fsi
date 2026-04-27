from IO3D import write_vector_3D
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(0)
np.set_printoptions(suppress=True)

def restrict(fine, Nx, Ny,Nz):
    """Restrict v to the coarse grid"""
    coarse  = np.zeros((Nx+2, Ny+2,Nz+2))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                coarse[i+1,j+1,k+1] = (
                    fine[2*i+1, 2*j+1, 2*k+2]+fine[2*i+2, 2*j+1, 2*k+2]+fine[2*i+1, 2*j+2, 2*k+2]+fine[2*i+2, 2*j+2, 2*k+2]+
                    fine[2*i+1, 2*j+1, 2*k+1]+fine[2*i+2, 2*j+1, 2*k+1]+fine[2*i+1, 2*j+2, 2*k+1]+fine[2*i+2, 2*j+2, 2*k+1])/8.0
    return coarse


def restrict_bc(fine, Nx, Ny, Nz):
    """Restrict boundary type to the coarse grid"""
    coarse = np.zeros((Nx+2, Ny+2,Nz+2))
    for j in range(Ny):
        for k in range(Nz):
            coarse[0, j+1, k+1] = max(fine[0, 2*j+1, 2*k+1], fine[0, 2*j+2, 2*k+1], fine[0, 2*j+1, 2*k+2], fine[0, 2*j+2, 2*k+2])
            coarse[Nx+1,j+1,k+1] = max(fine[2*Nx+1, 2*j+1, 2*k+1], fine[2*Nx+1, 2*j+2, 2*k+1], fine[2*Nx+1, 2*j+1, 2*k+2], fine[2*Nx+1, 2*j+2, 2*k+2])
    
    for i in range(Nx):
        for k in range(Nz):
            coarse[i+1, 0, k+1] = max(   fine[2*i+1, 0,      2*k+1], fine[2*i+2, 0,      2*k+1], fine[2*i+1, 0,      2*k+2], fine[2*i+2, 0,      2*k+2])
            coarse[i+1, Ny+1, k+1] = max(fine[2*i+1, 2*Ny+1, 2*k+1], fine[2*i+2, 2*Ny+1, 2*k+1], fine[2*i+1, 2*Ny+1, 2*k+2], fine[2*i+2, 2*Ny+1, 2*k+2])
    for i in range(Nx):
        for j in range(Ny):
            coarse[i+1, j+1, 0] = max(   fine[2*i+1, 2*j+1, 0],      fine[2*i+2, 2*j+1, 0],      fine[2*i+1, 2*j+2, 0],      fine[2*i+2, 2*j+2, 0])
            coarse[i+1, j+1, Nz+1] = max(fine[2*i+1, 2*j+1, 2*Nz+1], fine[2*i+2, 2*j+1, 2*Nz+1], fine[2*i+1, 2*j+2, 2*Nz+1], fine[2*i+2, 2*j+2, 2*Nz+1])
    return coarse

def interpolate(coarse, Nx, Ny, Nz):
    """interpolate v to the fine grid"""
    def L0(x): return 1.0 - x
    def L1(x): return x
    L = [L0, L1]
    
    def trilinear(x,y,z,f):
        sum = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sum += f[k*4+j*2+i]*L[i](x)*L[j](y)*L[k](z)
        return sum

    fine = np.zeros((2*Nx+2, 2*Ny+2,2*Nz+2))
    for i in range(0, Nx+1):
        for j in range(0, Ny+1):
            for k in range(0,Nz+1):
                f = [coarse[i, j, k],   coarse[i+1, j, k],   coarse[i, j+1, k],   coarse[i+1, j+1, k],
                     coarse[i, j, k+1], coarse[i+1, j, k+1], coarse[i, j+1, k+1], coarse[i+1, j+1, k+1]]
                fine[2*i,   2*j,   2*k]   = trilinear(0.25, 0.25, 0.25, f)
                fine[2*i,   2*j,   2*k+1] = trilinear(0.25, 0.25, 0.75, f)
                fine[2*i,   2*j+1, 2*k]   = trilinear(0.25, 0.75, 0.25, f)
                fine[2*i,   2*j+1, 2*k+1] = trilinear(0.25, 0.75, 0.75, f)
                fine[2*i+1, 2*j,   2*k]   = trilinear(0.75, 0.25, 0.25, f)
                fine[2*i+1, 2*j,   2*k+1] = trilinear(0.75, 0.25, 0.75, f)
                fine[2*i+1, 2*j+1, 2*k]   = trilinear(0.75, 0.75, 0.25, f)
                fine[2*i+1, 2*j+1, 2*k+1] = trilinear(0.75, 0.75, 0.75, f)
    return fine



def compute_multigrid_levels_3D(Nx, Ny, Nz, width, height, depth):
    dx,dy,dz = width/Nx, height/Ny, depth/Nz
    num_levels = 0
    level_dh = []
    level_dim = []
    while True:
        num_levels += 1
        level_dh.append((dx, dy,dz))
        level_dim.append((Nx, Ny, Nz))
        print(f"num_levels : {num_levels}, level_dh  : {dx:.4e}, {dy:.4e}, {dz:.4e}")
        print(f"num_levels : {num_levels}, level_dim : {Nx}, {Ny}, {Nz}")
        if Nx % 2 == 1 or Ny % 2 == 1 or Nz % 2 == 1:
            break
        if Nx * Ny * Nz < 65:
            break
        Nx, Ny, Nz = (Nx // 2, Ny // 2, Nz // 2)
        dx,dy,dz = (dx* 2.0, dy* 2.0, dz * 2.0)
    if num_levels <= 1:
        raise Exception("the problem is too small for the multigrid solver.")
    return num_levels, level_dh, level_dim

def compute_multigrid_levels_2D(Nx, Ny, width, height):
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

if __name__ == "__main__":
    Nx = 16
    Ny = 16
    Nz = 16
    coarse  = np.zeros((Nx+2  , Ny+2,   Nz+2))
    fine    = np.zeros((2*Nx+2, 2*Ny+2, 2*Nz+2))

    # for i in range(2*Nx+2):
    #     for j in range(2*Ny+1):
    #         fine[i,j] = (i + 100*j)


    for i in range(2*Nx+2):
        for j in range(2*Ny+2):
            for k in range(2*Nz+2):
                fine[i,j,k] = (i + 100*j + 10000*k)*1

    write_vector_3D(fine,'fine.txt')
    
    coarse = restrict(fine, Nx, Ny, Nz)
    write_vector_3D(coarse,'coarse.txt')
    
    coarse = restrict_bc(fine, Nx, Ny, Nz)
    write_vector_3D(coarse,'coarse.txt')

    for i in range(Nx+2):
        for j in range(Ny+2):
            for k in range(Ny+2):
                coarse[i,j,k] = i+j+k
    
    fine = interpolate(coarse, Nx, Ny,Nz)
    write_vector_3D(coarse,'coarse.txt')
    write_vector_3D(fine,'fine.txt')
    
    width = 1.0
    height = 1.0
    depth = 1.0
    compute_multigrid_levels_2D(Nx, Ny, width, height)
    compute_multigrid_levels_3D(Nx, Ny,Nz, width, height,depth)

