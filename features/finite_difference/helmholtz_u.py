from localtools import make_examples_helmholtz
from localtools import make_coordinates
from localtools import convergence_rates
from localtools import u_1, u_2, u_3,u_4
from localtools import NEUMANN, DIRICHLET

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def solve_helmholtz(Nx, Ny, max_iters, raw_u):

    dt = 0.000500
    mu = 0.01
    rho = 1.0
    u, du, f = make_examples_helmholtz(raw_u,dt=dt,mu=mu,rho=1.0)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    residual =          np.zeros((Nx+1, Ny+2))
    b =                 np.zeros((Nx+1, Ny+2))
    phi =               np.zeros((Nx+1, Ny+2))
    u_exact =           np.zeros((Nx+1, Ny+2))
    error =             np.zeros((Nx+1, Ny+2))
    boundary_type    =  np.zeros((Nx+1, Ny+2))
    boundary_values  =  np.zeros((Nx+1, Ny+2))

    # 上 $u_{i,-\frac{1}{2}},i\in[0,N_x]$
    # 下 $u_{i,N_y+\frac{1}{2}},i\in[0,N_x]$
    # 上下边界需包含左右角点
    for i in range(0, Nx+1):
        boundary_type[i][0]    = DIRICHLET
        boundary_type[i][Ny+1] = DIRICHLET

    # 左 $u_{0,j-\frac{1}{2}},j\in[0,N_y]$
    # 右 $u_{N_x,j-\frac{1}{2}},j\in[0,N_y]$
    for j in range(1, Ny+1):
        boundary_type[0][j]    = DIRICHLET
        boundary_type[Nx][j]   = DIRICHLET

    # 计算真实解和右端项
    for i in range(Nx+1):
        for j in range(Ny+2):
            u_exact[i][j] = u(dx*i, dy*j-0.5*dy)
            b[i][j] = f(dx*i, dy*j-0.5*dy)
            # u_exact[i][j] = u(X[i][j], Y[i][j])
            # b[i][j] = f(X[i][j], Y[i][j])
        
    # 计算边界上的值
    # 上 $u_{i,-\frac{1}{2}},i\in[0,N_x]$
    # 下
    for i in range(0, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(dx*i, 0)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](dx*i, 0)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(dx*i, 1)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](dx*i, 1)

    # 左
    # 右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j-0.5*dy)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j-0.5*dy)
        if boundary_type[Nx][j] == DIRICHLET:
            boundary_values[Nx][j] = u(1, dy*j-0.5*dy)
        if boundary_type[Nx][j] == NEUMANN:
            boundary_values[Nx][j] = du[0](1, dy*j-0.5*dy)


    for iter in range(max_iters):
        # 上下
        for i in range(0, Nx+1):
            if boundary_type[i][0] == DIRICHLET:
                phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
            if boundary_type[i][0] == NEUMANN:
                phi[i][0] = phi[i][1]-dy*boundary_values[i][0]
            if boundary_type[i][Ny+1] == DIRICHLET:
                phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
            if boundary_type[i][Ny+1] == NEUMANN:
                phi[i][Ny+1] = phi[i][Ny]+dy*boundary_values[i][Ny+1]
        
        # 左右
        for j in range(1, Ny+1):
            if boundary_type[0][j] == DIRICHLET:
                phi[0][j] = boundary_values[0][j]
            if boundary_type[0][j] == NEUMANN:
                phi_ghost = phi[1][j] - 2*dx*boundary_values[0][j]
                p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                p2 = (phi[1][j]+phi_ghost)/dx/dx+(phi[0][j+1] + phi[0][j-1])/dy/dy
                phi[0][j] = (b[0][j] + mu/rho*p2)/p1

                # phi[0][j] = ((phi[1][j]+phi_ghost)*dy*dy+(phi[0][j+1] +
                #              phi[0][j-1])*dx*dx-b[0][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)
            if boundary_type[Nx][j] == DIRICHLET:
                phi[Nx][j] = boundary_values[Nx][j]
            if boundary_type[Nx][j] == NEUMANN:
                phi_ghost = phi[Nx-1][j] + 2*dx*boundary_values[Nx][j]                
                p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                p2 = (phi_ghost+phi[Nx-1][j])/dx/dx+(phi[Nx][j+1] + phi[Nx][j-1])/dy/dy
                phi[Nx][j] = (b[Nx][j] + mu/rho*p2)/p1


        # print(phi[0][0],phi[Nx][0],phi[0][Ny+1],phi[Nx][Ny+1])

        # 迭代求解
        for i in range(1, Nx):
            for j in range(1, Ny+1):
                p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx+(phi[i][j+1] + phi[i][j-1])/dy/dy
                phi[i][j] = (b[i][j] + mu/rho*p2)/p1
        
        for i in range(1, Nx):
            for j in range(1, Ny+1):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx 
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = b[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)

        
        print(iter, np.sqrt(np.sum(residual**2)*dx*dy))
        if (np.sum(residual**2)*dx*dy< 1e-15): 
            break 

    for i in range(0, Nx+1):
        for j in range(1, Ny+1):
            error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sum(error**2)*dx*dy
    return error_norm_l2

solve_helmholtz(200, 200, max_iters=400, raw_u=u_2)

# def convergence_test(u):
#     E_list = []
#     h_list = []
#     N_list = []
#     for i in range(2, 6):
#         Nx = 8+4*i
#         Ny = 8+4*i
#         h = 1.0/Nx
#         E = solve_helmholtz(Nx, Ny, max_iters=40000, raw_u=u)
#         E_list.append(E)
#         h_list.append(h)
#         N_list.append(Nx)

#     R_list = convergence_rates(h_list, E_list)

#     return h_list, E_list, R_list


# print(convergence_test(u_4))
# print(convergence_test(u_3))
# print(convergence_test(u_2))
# print(convergence_test(u_1))
