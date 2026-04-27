from localtools import NEUMANN, DIRICHLET


import numpy as np
import matplotlib.pyplot as plt

def restrict(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse  = np.zeros((Nx+2  , Ny+2))
    for i in range(1,Nx+1):
        for j in range(1, Ny+1):
            coarse[i,j] = 0.25*(
                fine[2*i-1, 2*j-1]+fine[2*i, 2*j-1]+fine[2*i-1, 2*j]+fine[2*i, 2*j])
    return coarse

def restrict_bc(fine, Nx, Ny):
    """Restrict v to the coarse grid"""
    coarse  = np.zeros((Nx+2  , Ny+2))
    for i in range(1,Nx+1):
        for j in range(1, Ny+1):
            coarse[i,j] = max(fine[2*i-1, 2*j-1],fine[2*i, 2*j-1],fine[2*i-1, 2*j],fine[2*i, 2*j])
    return coarse

def interpolate(coarse, Nx, Ny):
    """interpolate v to the fine grid"""
    L0 = lambda x: 1.0 - x
    L1 = lambda x: x
    L = [L0, L1]
    fine    = np.zeros((2*Nx+2, 2*Ny+2))
    def bilinear(x, y, f):
        sum = 0.0
        for i in range(0,2):
            for j in range(0,2):
                sum += f[j*2+i]*L[i](x)*L[j](y)
        return sum
    for i in range(0,Nx+1):
        for j in range(0, Ny+1):
            f = [coarse[i,j],coarse[i+1,j],coarse[i,j+1],coarse[i+1,j+1]]
            fine[2*i,2*j] = bilinear(1.0/3.0, 1.0/3.0, f)
            fine[2*i,2*j+1] = bilinear(1.0/3.0, 2.0/3.0, f)
            fine[2*i+1,2*j] = bilinear(2.0/3.0, 1.0/3.0, f)
            fine[2*i+1,2*j+1] = bilinear(2.0/3.0, 2.0/3.0, f)
    return fine

if __name__ == "__main__":
    Nx = 4
    Ny = 4
    coarse  = np.zeros((Nx+2  , Ny+2))
    fine    = np.zeros((2*Nx+2, 2*Ny+2))

    for i in range(2*Nx+2):
        for j in range(2*Ny+2):
            fine[i,j] = (i + 100*j)*3


    for i in range(Nx+2):
        for j in range(Ny+2):
            coarse[i,j] = (i + 100*j)*3

    print(fine)
    print(coarse)

    coarse = restrict(fine, Nx, Ny)
    print(coarse)

    for i in range(Nx+2):
        for j in range(Ny+2):
            coarse[i,j] = (i + 100*j)*3
    
    fine = interpolate(coarse, Nx, Ny)
    print(fine)




