from localtools import NEUMANN, DIRICHLET


import numpy as np
import matplotlib.pyplot as plt

def restrict(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse  = np.zeros((Nx+1, Ny+2))
    for i in range(Nx+1):
        for j in range(1, Ny+1):
            coarse[i,j] = 0.5*(fine[2*i, 2*j]+fine[2*i, 2*j-1])
    return coarse

def restrict_bc(fine, Nx, Ny):
    """Restrict boundary type to the coarse grid"""
    coarse = np.zeros((Nx+1, Ny+2))
    for i in range(0,Nx+1): # 上下边界
        coarse[i][0] = fine[2*i, 0]
        coarse[i][Ny+1] = fine[2*i, 2*Ny+1]
    for j in range(1, Ny+1):
        coarse[0][j] = max(fine[0, 2*j-1],fine[0, 2*j])
        coarse[Nx][j] = max(fine[2*Nx, 2*j-1], fine[2*Nx, 2*j])
    return coarse

def interpolate(coarse, Nx, Ny):
    """interpolate v to the fine grid"""
    fine    = np.zeros((2*Nx+1, 2*Ny+2))
    for i in range(0,Nx+1):
        for j in range(0, Ny+1):
            a,b = coarse[i,j],coarse[i,j+1]
            c = 0.25*(b-a)
            fine[2*i,2*j] = a+c
            fine[2*i,2*j+1] = a+3*c

    for i in range(0,Nx):
        for j in range(0, Ny+1):
            fine[2*i+1,2*j] = 0.5*(fine[2*i,2*j] + fine[2*i+2,2*j])
            fine[2*i+1,2*j+1] = 0.5*(fine[2*i,2*j+1] + fine[2*i+2,2*j+1])

    return fine

if __name__ == "__main__":
    Nx = 4
    Ny = 4
    coarse  = np.zeros((Nx+1  , Ny+2))
    fine    = np.zeros((2*Nx+1, 2*Ny+2))

    for i in range(2*Nx+1):
        for j in range(2*Ny+2):
            fine[i,j] = (i + 100*j)


    for i in range(Nx+1):
        for j in range(Ny+2):
            coarse[i,j] = (i + 100*j)*3

    print(fine)
    print(coarse)

    coarse = restrict(fine, Nx, Ny)
    print(coarse)

    coarse = restrict_bc(fine, Nx, Ny)
    print(coarse)

    for i in range(Nx+1):
        for j in range(Ny+2):
            coarse[i,j] = (i + 100*j)*3
    
    fine = interpolate(coarse, Nx, Ny)
    print(fine)






