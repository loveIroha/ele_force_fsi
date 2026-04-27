

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

x, y = sym.symbols('x y')
u_1 = sym.sin(np.pi*x)*sym.sin(np.pi*y)+1
u_2 = (x*x-x)*(y*y-y)*sym.exp(x*y)


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


def fun_paint(X, Y, Z, title='title', x_axis='x axis', y_axis='y axis', z_axis='z axis', filename='file.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.view_init(elev=10., azim=-140)
    plt.show()
    plt.savefig(filename)


def fun(Nx=32, Ny=32, max_iters=100, u_1=u_1):

    # discretizing geometry
    u, du, f = make_example(u_1)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    coordinate_x = np.linspace(-0.5*dx, width+0.5*dx, Nx+2)
    coordinate_y = np.linspace(-0.5*dy, height+0.5*dy, Ny+2)

    Y, X = np.meshgrid(coordinate_x, coordinate_y)

    b = np.zeros((Nx+2, Ny+2))
    phi = np.zeros((Nx+2, Ny+2))
    u_exact = np.zeros((Nx+2, Ny+2))
    error = np.zeros((Nx+2, Ny+2))
    boundary_type = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        boundary_type[i][0] = NEUMANN
        boundary_type[i][Ny+1] = DIRICHLET

    for j in range(1, Ny+1):
        boundary_type[0][j] = DIRICHLET
        boundary_type[Nx+1][j] = DIRICHLET

    boundary_values = np.zeros((Nx+2, Ny+2))

    for i in range(Nx+2):
        for j in range(Ny+2):
            u_exact[i][j] = u(X[i][j], Y[i][j])
            b[i][j] = f(X[i][j], Y[i][j])

    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(X[i][0], 0)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](X[i][0], 0)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(X[i][Ny+1], 1)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](X[i][Ny+1], 1)

    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, Y[0][j])
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, Y[0][j])
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(1, Y[Nx+1][j])
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](1, Y[Nx+1][j])

    # fun_paint(X,Y,u_exact)

# (phi_i+1,j + phi_i-1,j - 2 phi_i,j)/dx/dx+(phi_i,j+1 + phi_i,j+1 - 2 phi_i,j)/dy/dy = b_ij

# dy*dy*(phi_i+1,j + phi_i-1,j)+dx*dx*(phi_i,j+1 + phi_i,j+1)-(2*dy*dy+2*dx*dx)phi_i,j=b_ij*dx*dx*dy*dy
# (dy*dy*(phi_i+1,j + phi_i-1,j)+dx*dx*(phi_i,j+1 + phi_i,j+1)-b_ij)/(2*dy*dy+2*dx*dx)=phi_i,j

    for iter in range(max_iters):
        # print(Nx,Ny,iter)
        for i in range(1, Nx+1):
            # 将它们写在一起
            # a = 1
            # b = 0
            phi[i][0] = float(boundary_type[i][0] == DIRICHLET)*(2*boundary_values[i][0] - phi[i][1]) \
                      + float(boundary_type[i][0] == NEUMANN) * (phi[i][1] - dx*boundary_values[i][0])
            phi[i][Ny+1] = float(boundary_type[i][Ny+1] == DIRICHLET)*(2*boundary_values[i][Ny+1]-phi[i][Ny]) \
                         + float(boundary_type[i][Ny+1] == NEUMANN)*(phi[i][Ny]+dx*boundary_values[i][Ny+1])
            # if boundary_type[i][0] == DIRICHLET:
            #     phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
            # if boundary_type[i][0] == NEUMANN:
            #     # phi[0]=phi[1]-dx*du(0.0)
            #     phi[i][0] = phi[i][1]-dx*boundary_values[i][0]
            # if boundary_type[i][Ny+1] == DIRICHLET:
            #     # phi[N+1]=2*u(1.0)-phi[N]
            #     phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
            # if boundary_type[i][Ny+1] == NEUMANN:
            #     phi[i][Ny+1] = phi[i][Ny]+dx*boundary_values[i][Ny+1]

        for j in range(1, Ny+1):
            if boundary_type[0][j] == DIRICHLET:
                # phi[0]=2*u(0.0)-phi[1]
                phi[0][j] = 2*boundary_values[0][j]-phi[1][j]
            if boundary_type[Nx+1][j] == DIRICHLET:
                # phi[N+1]=2*u(1.0)-phi[N]
                phi[Nx+1][j] = 2*boundary_values[Nx+1][j]-phi[Nx][j]
        # 迭代求解
        for i in range(1, Nx+1):
            for j in range(1, Ny+1):
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
    for i in range(2, 7):
        Nx = 2<<i
        Ny = 2<<i
        h = 1.0/Nx
        E = fun(Nx, Ny, max_iters=4000, u_1=u_2)
        E_list.append(E)
        h_list.append(h)
        N_list.append(Nx)

    # 计算收敛率

    def convergence_rates(h, E):
        """
        Given a sequence of discretization parameters in the list h,
        and corresponding errors in the list E,
        compute the convergence rate of two successive (h[i], E[i])
        and (h[i+1],E[i+1]) experiments, assuming the model E=C*h^r
        (for small enough h).
        """
        from math import log
        r = [log(E[i]/E[i-1])/log(h[i]/h[i-1])
             for i in range(1, len(h))]
        return r

    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list


print(fun2(u_2))
