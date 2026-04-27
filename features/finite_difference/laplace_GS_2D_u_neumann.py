

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from localtools import convergence_rates
from localtools import make_example
from localtools import u_1,u_2,u_3

DIRICHLET = 1
NEUMANN = 2


def fun(Nx=32, Ny=32, max_iters=100, u_1=u_1):

    # discretizing geometry
    u, du, f = make_example(u_1)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    coordinate_x = np.linspace(0, width, Nx+1)
    coordinate_y = np.linspace(-0.5*dy, height+0.5*dy, Ny+2)

    X = np.zeros((Nx+1, Ny+2))
    Y = np.zeros((Nx+1, Ny+2))

    for i in range(Nx+1):
        for j in range(Ny+2):
            X[i][j] = coordinate_x[i]
            Y[i][j] = coordinate_y[j]

    b =             np.zeros((Nx+1, Ny+2))
    phi =           np.zeros((Nx+1, Ny+2))
    u_exact =       np.zeros((Nx+1, Ny+2))
    error =         np.zeros((Nx+1, Ny+2))
    boundary_type = np.zeros((Nx+1, Ny+2))

    # 上下
    for i in range(0, Nx+1):
        boundary_type[i][0]    = DIRICHLET
        boundary_type[i][Ny+1] = NEUMANN

    # 左右
    for j in range(1, Ny+1):
        boundary_type[0][j]    = NEUMANN
        boundary_type[Nx][j]   = DIRICHLET

    boundary_values = np.zeros((Nx+1, Ny+2))

    for i in range(Nx+1):
        for j in range(Ny+2):
            u_exact[i][j] = u(X[i][j], Y[i][j])
            b[i][j] = f(X[i][j], Y[i][j])
    
    # 上下
    for i in range(0, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(X[i][0], 0)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](X[i][0], 0)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(X[i][Ny+1], 1)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](X[i][Ny+1], 1)
    
    # 左右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, Y[0][j])
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, Y[0][j])
        if boundary_type[Nx][j] == DIRICHLET:
            boundary_values[Nx][j] = u(1, Y[Nx][j])
        if boundary_type[Nx][j] == NEUMANN:
            boundary_values[Nx][j] = du[0](1, Y[Nx][j])


    for iter in range(max_iters):
        # 上下
        for i in range(0, Nx+1):
            if boundary_type[i][0] == DIRICHLET:
                phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
                # print(boundary_values[i][0], phi[i][0], phi[i][1])
            if boundary_type[i][0] == NEUMANN:
                phi[i][0] = phi[i][1]-dy*boundary_values[i][0]
            if boundary_type[i][Ny+1] == DIRICHLET:
                phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
                # print(phi[i][Ny+1], boundary_values[i][Ny+1],phi[i][Ny])
            if boundary_type[i][Ny+1] == NEUMANN:
                phi[i][Ny+1] = phi[i][Ny]+dy*boundary_values[i][Ny+1]
        
        # 左右
        for j in range(1, Ny+1):
            if boundary_type[0][j] == DIRICHLET:
                phi[0][j] = boundary_values[0][j]
                # print(boundary_values[0][j])
            if boundary_type[0][j] == NEUMANN:
                phi_ghost = phi[1][j] - 2*dx*boundary_values[0][j]
                phi[0][j] = ((phi[1][j]+phi_ghost)*dy*dy+(phi[0][j+1] +
                             phi[0][j-1])*dx*dx-b[0][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)
            if boundary_type[Nx][j] == DIRICHLET:
                phi[Nx][j] = boundary_values[Nx][j]
                # print(boundary_values[Nx][j])
            if boundary_type[Nx][j] == NEUMANN:
                phi_ghost = phi[Nx-1][j] + 2*dx*boundary_values[Nx][j]
                phi[Nx][j] = ((phi_ghost+phi[Nx-1][j])*dy*dy+(phi[Nx][j+1] +
                             phi[Nx][j-1])*dx*dx-b[Nx][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)


        # print(phi[0][0],phi[Nx][0],phi[0][Ny+1],phi[Nx][Ny+1])

        # 迭代求解
        for i in range(1, Nx):
            for j in range(1, Ny+1):
                phi[i][j] = ((phi[i+1][j]+phi[i-1][j])*dy*dy+(phi[i][j+1] +
                             phi[i][j-1])*dx*dx-b[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)

    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sum(error**2)*dx*dy
    return error_norm_l2


def convergence_test(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(2, 6):
        Nx = 8<<i
        Ny = 8<<i
        h = 1.0/Nx
        E = fun(Nx, Ny, max_iters=400000, u_1=u)
        E_list.append(E)
        h_list.append(h)
        N_list.append(Nx)

    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list


print(convergence_test(u_1))
print(convergence_test(u_2))
print(convergence_test(u_3))
