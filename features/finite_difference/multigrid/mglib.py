

import numpy as np
from localtools import NEUMANN, DIRICHLET

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
        # print(f"num_levels : {num_levels}, level_dh  : {dx:.4e}, {dy:.4e}")
        # print(f"num_levels : {num_levels}, level_dim : {Nx}, {Ny}")
        if Nx % 2 == 1 or Ny % 2 == 1:
            break
        if Nx * Ny < 17:
            break
        Nx, Ny = (Nx // 2, Ny // 2)
        dx, dy = (dx * 2, dy * 2)
    if num_levels <= 1:
        raise Exception("the problem is too small for the multigrid solver.")
    return num_levels, level_dh, level_dim


# compute_multigrid_levels_2D(64,64,1.0,1.0)


def restrict(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            coarse[i, j] = 0.25*(
                fine[2*i-1, 2*j-1]+fine[2*i, 2*j-1]+fine[2*i-1, 2*j]+fine[2*i, 2*j])
    return coarse


if __name__ == '__main__':
    Nx = 4
    Ny = 4
    coarse = np.zeros((Nx+2, Ny+2))
    fine = np.zeros((2*Nx+2, 2*Ny+2))
    for i in range(2*Nx+2):
        for j in range(2*Ny+2):
            fine[i, j] = (i + 100*j)
    coarse = restrict(fine, Nx, Ny)
    print(fine)
    print(coarse)

def compute_residual(phi, boundary_type, b, Nx, Ny, width, height):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        if boundary_type[i][0] == NEUMANN:
            phi[i][0] = phi[i][1]
        if boundary_type[i][Ny+1] == NEUMANN:
            phi[i][Ny+1] = phi[i][Ny]

    for j in range(1, Ny+1):
        if boundary_type[0][j] == NEUMANN:
            phi[0][j] = phi[1][j]
        if boundary_type[Nx+1][j] == NEUMANN:
            phi[Nx+1][j] = phi[Nx][j]

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
    return num_levels, level_dh, level_dim




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
