from localtools import make_examples_helmholtz,make_example
from localtools import make_coordinates
from localtools import convergence_rates
from localtools import u_1, u_2, u_3,u_4
from localtools import NEUMANN, DIRICHLET

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def solve_pressure(Nx=32, Ny=32, max_iters=100, raw_u=u_1):

    u, du, f = make_example(raw_u)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny


    residual =          np.zeros((Nx+2, Ny+2))
    b =                 np.zeros((Nx+2, Ny+2))
    phi =               np.zeros((Nx+2, Ny+2))
    u_exact =           np.zeros((Nx+2, Ny+2))
    error =             np.zeros((Nx+2, Ny+2))
    boundary_type    =  np.zeros((Nx+2, Ny+2))
    boundary_values  =  np.zeros((Nx+2, Ny+2))

    # 下
    for i in range(1, Nx+1):
        boundary_type[i][0]    = DIRICHLET
        boundary_type[i][Ny+1] = NEUMANN

    # 左
    # 右
    for j in range(1, Ny+1):
        boundary_type[0][j]    = NEUMANN
        boundary_type[Nx+1][j]   = DIRICHLET

    # 计算真实解和右端项
    for i in range(Nx+2):
        for j in range(Ny+2):
            u_exact[i][j] = u(dx*i-0.5*dx, dy*j-0.5*dy)
            b[i][j]       = f(dx*i-0.5*dx, dy*j-0.5*dy)
            # u_exact[i][j] = u(X[i][j], Y[i][j])
            # b[i][j] = f(X[i][j], Y[i][j])
        
    # 计算边界上的值
    # 上 $u_{i,-\frac{1}{2}},i\in[0,N_x]$
    # 下
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(dx*i-0.5*dx, 0)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](dx*i-0.5*dx, 0)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(dx*i-0.5*dx, 1)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](dx*i-0.5*dx, 1)

    # 左
    # 右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j-0.5*dy)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j-0.5*dy)
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(1, dy*j-0.5*dy)
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](1, dy*j-0.5*dy)


    for iter in range(max_iters):
        # print(Nx,Ny,iter)
        for i in range(1, Nx+1):
            if boundary_type[i][0] == DIRICHLET:
                phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
            if boundary_type[i][0] == NEUMANN:
                # phi[0]=phi[1]-dx*du(0.0)
                phi[i][0] = phi[i][1]-dx*boundary_values[i][0]
            if boundary_type[i][Ny+1] == DIRICHLET:
                # phi[N+1]=2*u(1.0)-phi[N]
                phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
            if boundary_type[i][Ny+1] == NEUMANN:
                phi[i][Ny+1] = phi[i][Ny]+dx*boundary_values[i][Ny+1]

        for j in range(1, Ny+1):
            if boundary_type[0][j] == DIRICHLET:
                # phi[0]=2*u(0.0)-phi[1]
                phi[0][j] = 2*boundary_values[0][j]-phi[1][j]
            if boundary_type[0][j] == NEUMANN:
                # phi[0]=phi[1]-dx*du(0.0)
                phi[0][j] = phi[1][j]-dy*boundary_values[0][j]
            if boundary_type[Nx+1][j] == DIRICHLET:
                # phi[N+1]=2*u(1.0)-phi[N]
                phi[Nx+1][j] = 2*boundary_values[Nx+1][j]-phi[Nx][j]
            if boundary_type[Nx+1][j] == NEUMANN:
                # phi[N+1]=phi[N]+dx*du(1.0)
                phi[Nx+1][j] = phi[Nx][j]+dy*boundary_values[Nx+1][j]

        # 迭代求解
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
                phi[i][j] = ((phi[i+1][j]+phi[i-1][j])*dy*dy+(phi[i][j+1] +
                             phi[i][j-1])*dx*dx-b[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx 
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = b[i][j] - p1 - p2
        
        if (np.sum(residual**2)*dx*dy< 1e-15): 
            print(iter,np.sum(residual**2)*dx*dy)
            break 

    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sum(error**2)*dx*dy
    from localtools import plot_p_3D
    plot_p_3D(phi,Nx,Ny)
    return error_norm_l2



def convergence_test(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(2, 6):
        Nx = 8+4*i
        Ny = 8+4*i
        h = 1.0/Nx
        E = solve_pressure(Nx, Ny, max_iters=40000, raw_u=u)
        E_list.append(E)
        h_list.append(h)
        N_list.append(Nx)

    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list


print(convergence_test(u_4))
print(convergence_test(u_3))
print(convergence_test(u_2))
print(convergence_test(u_1))
