from localtools import NEUMANN, DIRICHLET


import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(0)
np.set_printoptions(suppress=True)

def restrict(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse  = np.zeros((Nx+2, Ny+1))
    for i in range(1,Nx+1):
        for j in range(0, Ny+1):
            coarse[i,j] = 0.5*(fine[2*i, 2*j]+fine[2*i-1, 2*j])
    return coarse

def restrict_bc(fine, Nx, Ny):
    """Restrict boundary type to the coarse grid"""
    coarse = np.zeros((Nx+2, Ny+1))
    for i in range(1,Nx+1): # 上下边界
        coarse[i][0] = max(fine[2*i-1,0],fine[2*i,0])
        coarse[i][Ny] = max(fine[2*i-1,2*Ny],fine[2*i,2*Ny])
    for j in range(0, Ny+1):
        coarse[0][j] = fine[0,2*j]
        coarse[Nx+1][j] = fine[2*Nx+1,2*j]
    return coarse

def interpolate(coarse, Nx, Ny):
    """interpolate v to the fine grid"""
    fine    = np.zeros((2*Nx+2, 2*Ny+1))
    for i in range(0,Nx+1):
        for j in range(0, Ny+1):
            a,b = coarse[i,j],coarse[i+1,j]
            c = (b-a)*0.25
            fine[2*i,2*j] = a+c
            fine[2*i+1,2*j] = a+3*c

    for i in range(0, Nx+1):
        for j in range(0, Ny):
            fine[2*i,2*j+1] = 0.5*(fine[2*i,2*j] + fine[2*i,2*j+2])
            fine[2*i+1,2*j+1] = 0.5*(fine[2*i+1,2*j] + fine[2*i+1,2*j+2])

    return fine

if __name__ == "__main__":
    Nx = 4
    Ny = 4
    coarse  = np.zeros((Nx+2  , Ny+1))
    fine    = np.zeros((2*Nx+2, 2*Ny+1))

    # for i in range(2*Nx+2):
    #     for j in range(2*Ny+1):
    #         fine[i,j] = (i + 100*j)


    for i in range(Nx+2):
        for j in range(Ny+1):
            coarse[i,j] = (i + 100*j)*4

    # print(fine.T)
    print(coarse.T)

    # coarse = restrict(fine, Nx, Ny)
    # print(coarse.T)

    # for i in range(Nx+1):
    #     for j in range(Ny+2):
    #         coarse[i,j] = (i + 100*j)*3
    
    fine = interpolate(coarse, Nx, Ny)
    print(fine.T)

    # coarse = restrict_bc(fine, Nx, Ny)
    # print(coarse.T)




