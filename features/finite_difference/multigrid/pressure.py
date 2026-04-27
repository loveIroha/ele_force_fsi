from localtools import make_example
from localtools import convergence_rates
from localtools import NEUMANN, DIRICHLET
from localtools import plot_p_3D

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def generate_u_prime(boundary_type, boundary_values, Nx, Ny, width, height):
    dx = width/Nx
    dy = height/Ny
    phi = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
        if boundary_type[i][0] == NEUMANN:
            phi[i][0] = phi[i][1]-dx*boundary_values[i][0]
        if boundary_type[i][Ny+1] == DIRICHLET:
            phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
        if boundary_type[i][Ny+1] == NEUMANN:
            phi[i][Ny+1] = phi[i][Ny]+dx*boundary_values[i][Ny+1]
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = 2*boundary_values[0][j]-phi[1][j]
        if boundary_type[0][j] == NEUMANN:
            phi[0][j] = phi[1][j]-dy*boundary_values[0][j]
        if boundary_type[Nx+1][j] == DIRICHLET:
            phi[Nx+1][j] = 2*boundary_values[Nx+1][j]-phi[Nx][j]
        if boundary_type[Nx+1][j] == NEUMANN:
            phi[Nx+1][j] = phi[Nx][j]+dy*boundary_values[Nx+1][j]
    return phi


def generate_residual(phi, b, Nx, Ny, width, height):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+2, Ny+2))
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
            p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
            residual[i][j] = b[i][j] - p1 - p2
    return residual


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


def make_cell_centered_demo(Nx, Ny, width, height, raw_u, all_boundary_type):
    u, du, f = make_example(raw_u)
    dx = width/Nx
    dy = height/Ny
    b = np.zeros((Nx+2, Ny+2))
    u_exact = np.zeros((Nx+2, Ny+2))
    boundary_type = np.zeros((Nx+2, Ny+2))
    boundary_values = np.zeros((Nx+2, Ny+2))

    # 下
    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary_type[0]
        boundary_type[i][Ny+1] = all_boundary_type[1]

    # 左
    # 右
    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary_type[2]
        boundary_type[Nx+1][j] = all_boundary_type[3]

    # 计算真实解和右端项
    for i in range(Nx+2):
        for j in range(Ny+2):
            u_exact[i][j] = u(dx*i-0.5*dx, dy*j-0.5*dy)
            b[i][j] = f(dx*i-0.5*dx, dy*j-0.5*dy)
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
            boundary_values[i][Ny+1] = u(dx*i-0.5*dx, height)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](dx*i-0.5*dx, height)

    # 左
    # 右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j-0.5*dy)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j-0.5*dy)
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(width, dy*j-0.5*dy)
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](width, dy*j-0.5*dy)

    # 计算 u'
    # 计算 f - \Delta u'
    phi_prime = generate_u_prime(
        boundary_type, boundary_values, Nx, Ny, width, height)
    # np.savetxt('b_old.txt', b, fmt='%.4e', delimiter='  ')
    b = generate_residual(phi_prime, b, Nx, Ny, width, height)
    # np.savetxt('phi_prime.txt', phi_prime, fmt='%.4e', delimiter='  ')
    # np.savetxt('b_new.txt', b, fmt='%.4e', delimiter='  ')
    # np.savetxt('boundary_values.txt', boundary_values, fmt='%.4e', delimiter='  ')

    return phi_prime, b, u_exact, boundary_type


def solve_pressure(Nx, Ny, width, height, max_iters, raw_u, all_boundary_type):
    dx = width/Nx
    dy = height/Ny

    phi_prime, b, u_exact, boundary_type = make_cell_centered_demo(
        Nx, Ny, width, height, raw_u, all_boundary_type)

    residual = np.zeros((Nx+2, Ny+2))
    phi = np.zeros((Nx+2, Ny+2))
    error = np.zeros((Nx+2, Ny+2))


    # print(boundary_type)
    for iter in range(max_iters):
        phi = smooth(phi, boundary_type, b, Nx, Ny, width, height)

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = b[i][j] - p1 - p2

        if (np.sum(residual**2)*dx*dy < 1e-15):
            # print(iter,np.sum(residual**2)*dx*dy)
            break

    phi = phi + phi_prime

    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sqrt(np.sum(error**2)*dx*dy)
    plot_p_3D(phi, Nx, Ny, title="N = " + str(Nx)+" Contour Map")
    return error_norm_l2


def convergence_test(u):
    width = 2.0
    height = 2.0
    bcs = []
    for bc0 in range(1, 3):
        for bc1 in range(1, 3):
            for bc2 in range(1, 3):
                for bc3 in range(1, 3):
                    bcs.append([bc0, bc1, bc2, bc3])
    for bc in bcs:
        E_list = []
        h_list = []
        N_list = []
        for i in range(2, 6):
            Nx = 8+2*i
            Ny = 8+2*i
            h = 2.0/Nx
            E = solve_pressure(Nx, Ny, width, height,
                               max_iters=40000, raw_u=u, all_boundary_type=bc)
            E_list.append(E)
            h_list.append(h)
            N_list.append(Nx)

        R_list = convergence_rates(h_list, E_list)
        print(R_list)

    return h_list, E_list, R_list

# from localtools import u_1, u_2, u_3, u_4
# error = solve_pressure(16, 16, 1.0, 1.0, max_iters=40000, raw_u=u_1, all_boundary_type=[DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET])
if __name__ == "__main__":
    from localtools import u_1, u_2, u_3, u_4
    convergence_test(u_4)
    convergence_test(u_3)
    convergence_test(u_2)
    convergence_test(u_1)
