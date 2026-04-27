
from write_demo import NavierStokesDemo
import numpy as np

from localtools import NEUMANN, DIRICHLET
from localtools import plot_convergence_rate
from localtools import convergence_rates


# from write_demo import get_boundary_type_u, get_boundary_type_v
# from multigrid import get_boundary_values_u, get_boundary_values_v

# NOTE: In fact, there is no neccessary to assign values on the bottom and top edge


def make_tentitive_u_b(un, f, Nx, Ny, dt, rho):
    ub = np.zeros((Nx+1, Ny+2))
    for i in range(0, Nx+1):
        for j in range(0, Ny+2):
            ub[i][j] = un[i][j]/dt+f[i][j]/rho

    return ub


# NOTE: In fact, there is no neccessary to assign values on the left and right edge
def make_tentitive_v_b(vn, f, Nx, Ny, dt, rho):
    vb = np.zeros((Nx+2, Ny+1))
    for i in range(0, Nx+2):
        for j in range(0, Ny+1):
            vb[i][j] = vn[i][j]/dt+f[i][j]/rho

    return vb


NS_EPSILON = 1e-15


def generate_u_prime(boundary_type, boundary_values, ub, Nx, Ny, width, height, dt, rho, mu):
    phi = np.zeros((Nx+1, Ny+2))
    dx = width/Nx
    dy = height/Ny

    # 左右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = boundary_values[0][j]
        if boundary_type[0][j] == NEUMANN:
            phi_ghost = phi[1][j] - 2*dx*boundary_values[0][j]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[1][j]+phi_ghost)/dx/dx + \
                (phi[0][j+1] + phi[0][j-1])/dy/dy
            phi[0][j] = (ub[0][j] + mu/rho*p2)/p1
        if boundary_type[Nx][j] == DIRICHLET:
            phi[Nx][j] = boundary_values[Nx][j]
        if boundary_type[Nx][j] == NEUMANN:
            phi_ghost = phi[Nx-1][j] + 2.0*dx*boundary_values[Nx][j]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi_ghost+phi[Nx-1][j])/dx/dx + \
                (phi[Nx][j+1] + phi[Nx][j-1])/dy/dy
            phi[Nx][j] = (ub[Nx][j] + mu/rho*p2)/p1

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

    return phi


def generate_v_prime(boundary_type, boundary_values, vb, Nx, Ny, width, height, dt, rho, mu):
    phi = np.zeros((Nx+2, Ny+1))
    dx = width/Nx
    dy = height/Ny
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            phi[i][0] = boundary_values[i][0]
        if boundary_type[i][0] == NEUMANN:
            ghost_phi = phi[i][1]-2.0*dy*boundary_values[i][0]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][0]+phi[i-1][0])/dx / \
                dx+(phi[i][1] + ghost_phi)/dy/dy
            phi[i][0] = (vb[i][0] + mu/rho*p2)/p1
        if boundary_type[i][Ny] == DIRICHLET:
            phi[i][Ny] = boundary_values[i][Ny]
        if boundary_type[i][Ny] == NEUMANN:
            ghost_phi = phi[i][Ny-1]+2.0*dy*boundary_values[i][Ny]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][Ny]+phi[i-1][Ny])/dx / \
                dx+(ghost_phi + phi[i][Ny-1])/dy/dy
            phi[i][Ny] = (vb[i][Ny] + mu/rho*p2)/p1

    for j in range(0, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = 2*boundary_values[0][j]-phi[1][j]
        if boundary_type[0][j] == NEUMANN:
            phi[0][j] = phi[1][j]-dy*boundary_values[0][j]
        if boundary_type[Nx+1][j] == DIRICHLET:
            phi[Nx+1][j] = 2*boundary_values[Nx+1][j]-phi[Nx][j]
        if boundary_type[Nx+1][j] == NEUMANN:
            phi[Nx+1][j] = phi[Nx][j]+dy*boundary_values[Nx+1][j]

    np.savetxt("phi.txt", phi)
    return phi


def smooth_u(phi, boundary_type, ub, Nx, Ny, width, height, dt, rho, mu):
    dx = width/Nx
    dy = height/Ny
    # 上下
    for i in range(0, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            phi[i][0] = -phi[i][1]
        if boundary_type[i][0] == NEUMANN:
            phi[i][0] = phi[i][1]
        if boundary_type[i][Ny+1] == DIRICHLET:
            phi[i][Ny+1] = -phi[i][Ny]
        if boundary_type[i][Ny+1] == NEUMANN:
            phi[i][Ny+1] = phi[i][Ny]

    # 左右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = 0.0
        if boundary_type[0][j] == NEUMANN:
            phi_ghost = phi[1][j]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[1][j]+phi_ghost)/dx/dx + \
                (phi[0][j+1] + phi[0][j-1])/dy/dy
            phi[0][j] = (ub[0][j] + mu/rho*p2)/p1
        if boundary_type[Nx][j] == DIRICHLET:
            phi[Nx][j] = 0.0
        if boundary_type[Nx][j] == NEUMANN:
            phi_ghost = phi[Nx-1][j]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi_ghost+phi[Nx-1][j])/dx/dx + \
                (phi[Nx][j+1] + phi[Nx][j-1])/dy/dy
            phi[Nx][j] = (ub[Nx][j] + mu/rho*p2)/p1

    # 迭代求解
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx + \
                (phi[i][j+1] + phi[i][j-1])/dy/dy
            phi[i][j] = (ub[i][j] + mu/rho*p2)/p1

    return phi


def smooth_v(phi, boundary_type, vb, Nx, Ny, width, height, dt, rho, mu):
    dx = width/Nx
    dy = height/Ny
    # 左右边界
    for j in range(0, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = -phi[1][j]
        if boundary_type[0][j] == NEUMANN:
            phi[0][j] = phi[1][j]
        if boundary_type[Nx+1][j] == DIRICHLET:
            phi[Nx+1][j] = -phi[Nx][j]
        if boundary_type[Nx+1][j] == NEUMANN:
            phi[Nx+1][j] = phi[Nx][j]

    # 上下边界
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            phi[i][0] = 0.0
        if boundary_type[i][0] == NEUMANN:
            ghost_phi = phi[i][1]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][0]+phi[i-1][0])/dx / \
                dx+(phi[i][1] + ghost_phi)/dy/dy
            phi[i][0] = (vb[i][0] + mu/rho*p2)/p1
        if boundary_type[i][Ny] == DIRICHLET:
            phi[i][Ny] = 0.0
        if boundary_type[i][Ny] == NEUMANN:
            ghost_phi = phi[i][Ny-1]
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][Ny]+phi[i-1][Ny])/dx / \
                dx+(ghost_phi + phi[i][Ny-1])/dy/dy
            phi[i][Ny] = (vb[i][Ny] + mu/rho*p2)/p1

    # 迭代求解
    for i in range(1, Nx+1):
        for j in range(1, Ny):
            p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
            p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx + \
                (phi[i][j+1] + phi[i][j-1])/dy/dy
            phi[i][j] = (vb[i][j] + mu/rho*p2)/p1
    return phi


def generate_u_residual(phi, ub, boundary_type, boundary_values, Nx, Ny, width, height, dt, rho, mu):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+1, Ny+2))
    # 左右
    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            phi[0][j] = boundary_values[0][j]
        if boundary_type[0][j] == NEUMANN:
            phi_ghost = phi[1][j] - 2*dx*boundary_values[0][j]
            p1 = (phi_ghost-2*phi[0][j]+phi[1][j])/dx/dx
            p2 = (phi[0][j-1]-2*phi[0][j]+phi[0][j+1])/dy/dy
            residual[0][j] = ub[0][j] - phi[0][j]/dt + mu/rho*(p1+p2)
        if boundary_type[Nx][j] == DIRICHLET:
            phi[Nx][j] = boundary_values[Nx][j]
        if boundary_type[Nx][j] == NEUMANN:
            phi_ghost = phi[Nx-1][j] + 2.0*dx*boundary_values[Nx][j]
            p1 = (phi[Nx-1][j]-2*phi[Nx][j]+phi_ghost)/dx/dx
            p2 = (phi[Nx][j-1]-2*phi[Nx][j]+phi[Nx][j+1])/dy/dy
            residual[Nx][j] = ub[Nx][j] - phi[Nx][j]/dt + mu/rho*(p1+p2)
    for i in range(1, Nx):
        for j in range(1, Ny+1):
            p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
            p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
            residual[i][j] = ub[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)
    return residual


def generate_v_residual(phi, vb, boundary_type, boundary_values, Nx, Ny, width, height, dt, rho, mu):
    dx = width/Nx
    dy = height/Ny
    residual = np.zeros((Nx+2, Ny+1))

    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            phi[i][0] = boundary_values[i][0]
        if boundary_type[i][0] == NEUMANN:
            ghost_phi = phi[i][1]-2.0*dy*boundary_values[i][0]
            p1 = (phi[i-1][0]-2*phi[i][0]+phi[i+1][0])/dx/dx
            p2 = (ghost_phi-2*phi[i][0]+phi[i][1])/dy/dy
            residual[i][0] = vb[i][0] - phi[i][0]/dt + mu/rho*(p1+p2)
        if boundary_type[i][Ny] == DIRICHLET:
            phi[i][Ny] = boundary_values[i][Ny]
        if boundary_type[i][Ny] == NEUMANN:
            ghost_phi = phi[i][Ny-1]+2.0*dy*boundary_values[i][Ny]
            p1 = (phi[i-1][Ny]-2*phi[i][Ny]+phi[i+1][Ny])/dx/dx
            p2 = (phi[i][Ny-1]-2*phi[i][Ny]+ghost_phi)/dy/dy
            residual[i][Ny] = vb[i][Ny] - phi[i][Ny]/dt + mu/rho*(p1+p2)

    for i in range(1, Nx+1):
        for j in range(1, Ny):
            p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
            p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
            residual[i][j] = vb[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)
    return residual


# 输入：右端项 ub, 初解 un,
#      问题参数 rho, mu,
#      离散参数 Nx, Ny, dt
# 返回数值解 u_
def solve_helmholtz_u(un, ub, mu, rho, width, height, boundary_type, boundary_values, Nx, Ny, dt, max_iters):
    dx = width/Nx
    dy = height/Ny
    phi = np.copy(un)
    residual = np.zeros((Nx+1, Ny+2))
    # boundary_values = np.zeros((Nx+1, Ny+2))
    # 生成 u' 和 ub'
    phi_prime = generate_u_prime(
        boundary_type, boundary_values, ub, Nx, Ny, width, height, dt, rho, mu)
    ub = generate_u_residual(phi_prime, ub, boundary_type,
                             boundary_values, Nx, Ny, width, height, dt, rho, mu)

    for iter in range(max_iters):
        phi = smooth_u(phi, boundary_type, ub, Nx,
                       Ny, width, height, dt, rho, mu)

        for i in range(1, Nx):
            for j in range(1, Ny+1):
                # p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                # p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx + \
                #      (phi[i][j+1]+phi[i][j-1])/dy/dy
                # residual[i][j] = p1*phi[i][j] - (ub[i][j] + mu/rho*p2)
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = ub[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)

        if (np.sum(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.sum(residual**2)*dx*dy)
            break

    phi = phi + phi_prime
    return phi, residual




def solve_helmholtz_v(vn, vb,
                      mu, rho, width, height,
                      boundary_type, boundary_values,
                      Nx, Ny, dt, max_iters):

    dx = width/Nx
    dy = height/Ny

    phi = np.copy(vn)
    residual = np.zeros((Nx+2, Ny+1))
    # boundary_values = np.zeros((Nx+2, Ny+1))

    # 生成 v' 和 vb'
    phi_prime = generate_v_prime(
        boundary_type, boundary_values, vb, Nx, Ny, width, height, dt, rho, mu)
    vb = generate_v_residual(phi_prime, vb,
                             boundary_type, boundary_values, Nx, Ny, width, height, dt, rho, mu)

    for iter in range(max_iters):
        phi = smooth_v(phi, boundary_type, vb, Nx,
                       Ny, width, height, dt, rho, mu)

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = vb[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)

        if (np.sum(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.sum(residual**2)*dx*dy)
            break

    phi = phi + phi_prime
    return phi, residual


def solve_momentum(Nx=24, Ny=24, Nt=100, width=1.0, height=1.0, T=1.0, rho=1.0, mu=1.0):
    all_boundary_type = [NEUMANN, NEUMANN, DIRICHLET, DIRICHLET]
    demo_nv = NavierStokesDemo(
        Nx, Ny, Nt, width, height, T, rho, mu, all_boundary_type)
    uns, vns, f1ns, f2ns, ubvns, vbvns, btu, btv = demo_nv.generate_data()

    Nx = demo_nv.Nx
    Ny = demo_nv.Ny
    Nt = demo_nv.Nt

    dx = demo_nv.dx
    dy = demo_nv.dy
    dt = demo_nv.dt

    width = demo_nv.width
    height = demo_nv.height
    T = demo_nv.T

    rho = demo_nv.rho
    mu = demo_nv.mu

    un = uns[0]
    vn = vns[0]
    for i in range(1, Nt+1):
        t = i*dt
        print("NO. ", i, ", time: ", t)
        ub = make_tentitive_u_b(un, f1ns[i], Nx, Ny, dt, rho)
        vb = make_tentitive_v_b(vn, f2ns[i], Nx, Ny, dt, rho)

        u_, residual_u_ = solve_helmholtz_u(
            un, ub, mu, rho, width, height, btu, ubvns[i], Nx, Ny, dt, 10000)
        v_, residual_v_ = solve_helmholtz_v(
            vn, vb, mu, rho, width, height, btv, vbvns[i], Nx, Ny, dt, 10000)
        un = np.copy(u_)
        vn = np.copy(v_)

    error_u = np.sqrt(np.sum((u_ - uns[Nt])**2)*demo_nv.dx*demo_nv.dy)
    error_v = np.sqrt(np.sum((v_ - vns[Nt])**2)*demo_nv.dx*demo_nv.dy)

    print(error_u, error_v)
    return error_u


if __name__ == "__main__":
    EE_list = []
    tt_list = []
    RR_list = []
    width_2 = 1.0
    height_2 = 1.0
    T_2 = 1.0

    rho_2 = 1.0
    mu_2 = 0.01
    for i in range(4, 9):
        Nx = 8+4*i
        Ny = 8+4*i
        E_list = []
        t_list = []
        # 固定网格步长，计算不同时间步长的误差
        for j in range(1, 5):
            Nt = 1 << j
            print("Nx Ny Nt : ", Nx, Ny, Nt)
            E = solve_momentum(Nx, Ny, Nt, width=width_2,
                               height=height_2, T=T_2, rho=rho_2, mu=mu_2)
            E_list.append(E)
            t_list.append(T_2/Nt)

        R_list = convergence_rates(t_list, E_list)
        EE_list.append(E_list)
        tt_list.append(t_list)
        RR_list.append(R_list)
        print(E_list)

    plot_convergence_rate(tt_list, EE_list, RR_list)
