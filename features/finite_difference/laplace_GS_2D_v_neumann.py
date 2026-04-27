

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from localtools import convergence_rates

x, y = sym.symbols('x y')
u_1 = sym.sin(np.pi*x)*sym.sin(np.pi*y)+1
u_2 = (x*x-x)*(y*y-y)*sym.exp(x*y)+1
u_3 = (x*x-x)**2*(y*y-y)**2*sym.exp(x*y)+1


def laplace(u):
    return sym.diff(u, x, 2)+sym.diff(u, y, 2)


def grad(u):
    return sym.diff(u, x, 1), sym.diff(u, y, 1)


def make_example(u):
    lambda_u = sym.lambdify([x, y], u, 'numpy')
    lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
    lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
    lambda_laplace_u = sym.lambdify([x, y], laplace(u), 'numpy')
    return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u


DIRICHLET = 1
NEUMANN = 2


def fun(Nx=32, Ny=32, max_iters=100, u_1=u_1):

    # discretizing geometry
    u, du, f = make_example(u_1)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    coordinate_x = np.linspace(-0.5*dx, width+0.5*dx, Nx+2)
    coordinate_y = np.linspace(0, height, Ny+1)

    X = np.zeros((Nx+2, Ny+1))
    Y = np.zeros((Nx+2, Ny+1))

    for i in range(Nx+2):
        for j in range(Ny+1):
            X[i][j] = coordinate_x[i]
            Y[i][j] = coordinate_y[j]

    b = np.zeros((Nx+2, Ny+1))
    phi = np.zeros((Nx+2, Ny+1))
    u_exact = np.zeros((Nx+2, Ny+1))
    error = np.zeros((Nx+2, Ny+1))
    boundary_type = np.zeros((Nx+2, Ny+1))
    # 上下
    for i in range(1, Nx+1):
        boundary_type[i][0] = NEUMANN
        boundary_type[i][Ny] = NEUMANN

    # 左右
    for j in range(0, Ny+1):
        boundary_type[0][j] = NEUMANN
        boundary_type[Nx+1][j] = DIRICHLET

    boundary_values = np.zeros((Nx+2, Ny+1))

    for i in range(Nx+2):
        for j in range(Ny+1):
            u_exact[i][j] = u(X[i][j], Y[i][j])
            b[i][j] = f(X[i][j], Y[i][j])

    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(X[i][0], 0)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](X[i][0], 0)
        if boundary_type[i][Ny] == DIRICHLET:
            boundary_values[i][Ny] = u(X[i][Ny], 1)
        if boundary_type[i][Ny] == NEUMANN:
            boundary_values[i][Ny] = du[1](X[i][Ny], 1)

    for j in range(0, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, Y[0][j])
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, Y[0][j])
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(1, Y[Nx+1][j])
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](1, Y[Nx+1][j])


    for iter in range(max_iters):
        # 左右边界
        for j in range(0, Ny+1):
            if boundary_type[0][j] == DIRICHLET:
                phi[0][j] = 2*boundary_values[0][j]-phi[1][j]
            if boundary_type[0][j] == NEUMANN:
                phi[0][j] = phi[1][j]-dy*boundary_values[0][j]
            if boundary_type[Nx+1][j] == DIRICHLET:
                phi[Nx+1][j] = 2*boundary_values[Nx+1][j]-phi[Nx][j]
            if boundary_type[Nx+1][j] == NEUMANN:
                phi[Nx+1][j] = phi[Nx][j]+dy*boundary_values[Nx+1][j]

        # 上下边界
        for i in range(1, Nx+1):
            if boundary_type[i][0] == DIRICHLET:
                phi[i][0] = boundary_values[i][0]
            if boundary_type[i][0] == NEUMANN:
                ghost_phi = phi[i][1]-2.0*dy*boundary_values[i][0]
                phi[i][0] = ((phi[i-1][0]+phi[i+1][0])*dy*dy+(ghost_phi +
                             phi[i][1])*dx*dx-b[i][0]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)
            if boundary_type[i][Ny] == DIRICHLET:
                phi[i][Ny] = boundary_values[i][Ny]
            if boundary_type[i][Ny] == NEUMANN:
                ghost_phi = phi[i][Ny-1]+2.0*dy*boundary_values[i][Ny]
                phi[i][Ny] = ((phi[i-1][Ny]+phi[i+1][Ny])*dy*dy+(phi[i][Ny-1] +
                             ghost_phi)*dx*dx-b[i][Ny]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)



        # 迭代求解
        for i in range(1, Nx+1):
            for j in range(1, Ny):
                phi[i][j] = ((phi[i+1][j]+phi[i-1][j])*dy*dy+(phi[i][j+1] +
                             phi[i][j-1])*dx*dx-b[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)

    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sum(error**2)*dx*dy
    return error_norm_l2


def fun2(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(2, 6):
        Nx = 8+4*i
        Ny = 8+4*i
        h = 1.0/Nx
        E = fun(Nx, Ny, max_iters=40000, u_1=u)
        E_list.append(E)
        h_list.append(h)
        N_list.append(Nx)

    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list


print(fun2(u_1))
print(fun2(u_2))
print(fun2(u_3))
