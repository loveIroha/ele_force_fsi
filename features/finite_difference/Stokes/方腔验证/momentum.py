from localtools import NEUMANN, DIRICHLET
from localtools import convergence_rates
from localtools import plot_convergence_rate

from rhs import make_tentitive_v_b
from rhs import make_tentitive_u_b
from rhs import make_pressure_b
from rhs import correct_u, correct_v

import sympy as sym
import numpy as np


NS_EPSILON = 1e-15
T = 0.1

class NavierStokesDemo:

    x, y, t = sym.symbols('x y t')

    dt = 0.2
    Nx = 64
    Ny = 64

    width = 1.0
    height = 1.0

    dx = width  / Nx
    dy = height / Ny

    def __init__(self,rho = 1.0, mu = 1.0):
        self.rho = rho
        self.mu = mu

        # TODO: 问题参数和求解参数
        # 构造真实解
        x, y, t = self.x, self.y, self.t
        
        u = - sym.exp(t)*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256
        v = sym.exp(t)*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)/256
        p = sym.exp(t)*((x*x-x)*(y*y-y))

        u = sym.exp(t)*sym.sin(np.pi*x)**2*sym.sin(2*np.pi*y)
        v = -sym.exp(t)*sym.sin(2*np.pi*x)*sym.sin(np.pi*y)**2
        p = sym.exp(t)*((x*x-x)*(y*y-y))

        # u = 20*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)*t
        # v = 20*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)*t
        # 右端项
        (self.u, self.v, self.p), (self.du, self.dv, self.dp), (self.f1, self.f2) = \
            self.make_examples_NS_2D(u, v, p)

    # 1. 右端项 f1 f2
    # 2. 真实解 u v p
    # 3. 梯度 （u_x u_y, v_x v_y)
    def make_examples_NS_2D(self, u, v, p):
        mu=self.mu
        rho=self.rho
        x, y, t = self.x, self.y, self.t
        # 生成右端项 f1 和 f2
        f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) +
                                         sym.diff(u, y, 2)) + sym.diff(p, x, 1)
        f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) +
                                         sym.diff(v, y, 2)) + sym.diff(p, y, 1)
        u_ = sym.lambdify([x, y, t], u, 'numpy')
        v_ = sym.lambdify([x, y, t], v, 'numpy')
        p_ = sym.lambdify([x, y, t], p, 'numpy')
        f1_ = sym.lambdify([x, y, t], f1, 'numpy')
        f2_ = sym.lambdify([x, y, t], f2, 'numpy')
        dudx = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
        dudy = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
        dvdx = sym.lambdify([x, y], sym.diff(v, x, 1), 'numpy')
        dvdy = sym.lambdify([x, y], sym.diff(v, y, 1), 'numpy')
        dpdx = sym.lambdify([x, y], sym.diff(p, x, 1), 'numpy')
        dpdy = sym.lambdify([x, y], sym.diff(p, y, 1), 'numpy')
        return (u_, v_, p_), ((dudx, dudy), (dvdx, dvdy), (dpdx, dpdy)), (f1_, f2_)


# 输入：右端项 ub, 初解 un,
#      问题参数 rho, mu,
#      离散参数 Nx, Ny, dt
# 返回数值解 u_

def solve_helmholtz_u(un, ub,
                      mu=0.01, rho=1.0, width=1.0, height=1.0,
                      all_boundary=(DIRICHLET, DIRICHLET,
                                    DIRICHLET, DIRICHLET),
                      Nx=32, Ny=32, dt=100, max_iters=100):

    dx = width/Nx
    dy = height/Ny

    phi = np.copy(un)

    residual = np.zeros((Nx+1, Ny+2))
    boundary_type = np.zeros((Nx+1, Ny+2))
    boundary_values = np.zeros((Nx+1, Ny+2))

    # 上 $u_{i,-\frac{1}{2}},i\in[0,N_x]$
    # 下 $u_{i,N_y+\frac{1}{2}},i\in[0,N_x]$
    # 上下边界需包含左右角点
    for i in range(0, Nx+1):
        boundary_type[i][0] = all_boundary[0]
        boundary_type[i][Ny+1] = all_boundary[1]

    # 左 $u_{0,j-\frac{1}{2}},j\in[0,N_y]$
    # 右 $u_{N_x,j-\frac{1}{2}},j\in[0,N_y]$
    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary[2]
        boundary_type[Nx][j] = all_boundary[3]

    for iter in range(max_iters):
        # 上下
        for i in range(0, Nx+1):
            if boundary_type[i][0] == DIRICHLET:
                phi[i][0] = 2*boundary_values[i][0]-phi[i][1]
                # phi[i][0] = 2.0 - phi[i][1]
            if boundary_type[i][0] == NEUMANN:
                phi[i][0] = phi[i][1]-dy*boundary_values[i][0]
            if boundary_type[i][Ny+1] == DIRICHLET:
                # phi[i][Ny+1] = 2*boundary_values[i][Ny+1]-phi[i][Ny]
                phi[i][Ny+1] = 2.0 - phi[i][Ny]
            if boundary_type[i][Ny+1] == NEUMANN:
                phi[i][Ny+1] = phi[i][Ny]+dy*boundary_values[i][Ny+1]

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
                phi_ghost = phi[Nx-1][j] + 2*dx*boundary_values[Nx][j]
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

        for i in range(1, Nx):
            for j in range(1, Ny+1):
                p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx + \
                     (phi[i][j+1]+phi[i][j-1])/dy/dy
                residual[i][j] = p1*phi[i][j] - (ub[i][j] + mu/rho*p2)
        
        if (np.max(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.max(residual**2)*dx*dy)
            break

    return phi, residual


def solve_helmholtz_v(vn,vb,
                      mu=1.0, rho=1.0, width=1.0, height=1.0,
                      all_boundary=(DIRICHLET, DIRICHLET,
                                    DIRICHLET, DIRICHLET),
                      Nx=32, Ny=32, dt=100, max_iters=100):

    dx = width/Nx
    dy = height/Ny

    phi = np.copy(vn)
    residual = np.zeros((Nx+2, Ny+1))
    boundary_type = np.zeros((Nx+2, Ny+1))
    boundary_values = np.zeros((Nx+2, Ny+1))

    # 上
    # 下
    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary[0]
        boundary_type[i][Ny] = all_boundary[1]

    # 左
    # 右
    # 左右边界需包含左右角点
    for j in range(0, Ny+1):
        boundary_type[0][j] = all_boundary[2]
        boundary_type[Nx+1][j] = all_boundary[3]

    for iter in range(max_iters):
        # 上下边界
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

        # print(phi[0][0],phi[Nx][0],phi[0][Ny+1],phi[Nx][Ny+1])

        # 迭代求解
        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = 1/dt+2.0*mu/rho*(1.0/dx/dx+1.0/dy/dy)
                p2 = (phi[i+1][j]+phi[i-1][j])/dx/dx + \
                    (phi[i][j+1] + phi[i][j-1])/dy/dy
                phi[i][j] = (vb[i][j] + mu/rho*p2)/p1

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = vb[i][j] - phi[i][j]/dt + mu/rho*(p1+p2)

        if (np.max(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.max(residual**2)*dx*dy)
            break

    return phi, residual


def solve_poisson_p(pn, pb,
                    width=1.0, height=1.0,
                    all_boundary=(NEUMANN, NEUMANN,
                                  NEUMANN, NEUMANN),
                    Nx=32, Ny=32, max_iters=100):

    dx = width/Nx
    dy = height/Ny

    residual = np.zeros((Nx+2, Ny+2))
    phi = np.copy(pn)
    boundary_type = np.zeros((Nx+2, Ny+2))
    boundary_values = np.zeros((Nx+2, Ny+2))

    # 下
    # 上
    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary[0]
        boundary_type[i][Ny+1] = all_boundary[1]

    # 左
    # 右
    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary[2]
        boundary_type[Nx+1][j] = all_boundary[3]

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
                             phi[i][j-1])*dx*dx-pb[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)

        for i in range(1, Nx+1):
            for j in range(1, Ny):
                p1 = (phi[i-1][j]-2*phi[i][j]+phi[i+1][j])/dx/dx
                p2 = (phi[i][j-1]-2*phi[i][j]+phi[i][j+1])/dy/dy
                residual[i][j] = pb[i][j] - p1 - p2

        if (np.max(residual**2)*dx*dy < 1e-15):
            # print(iter, np.max(residual**2)*dx*dy)
            break
    return phi, residual




demo_nv = NavierStokesDemo()
def test_momentum(dt, Nx, Ny):

    demo_nv.dt = dt
    demo_nv.Nx = Nx
    demo_nv.Ny = Ny
    demo_nv.dx = demo_nv.width/Nx
    demo_nv.dy = demo_nv.height/Ny

    t = 0
    u = np.zeros((demo_nv.Nx+1, demo_nv.Ny+2))
    v = np.zeros((demo_nv.Nx+2, demo_nv.Ny+1))

    un = np.zeros((demo_nv.Nx+1, demo_nv.Ny+2))
    vn = np.zeros((demo_nv.Nx+2, demo_nv.Ny+1))

    pn = np.zeros((demo_nv.Nx+2, demo_nv.Ny+2))

    for iter in range(100000000):

        t += demo_nv.dt

        f1n = np.zeros((demo_nv.Nx+1, demo_nv.Ny+2))
        f2n = np.zeros((demo_nv.Nx+2, demo_nv.Ny+1))

        ub = make_tentitive_u_b(un, f1n, Nx=demo_nv.Nx,
                                Ny=demo_nv.Ny, dt=demo_nv.dt, rho=demo_nv.rho)
        vb = make_tentitive_v_b(vn, f2n, Nx=demo_nv.Nx,
                                Ny=demo_nv.Ny, dt=demo_nv.dt, rho=demo_nv.rho)

        u_, residual_u_ = solve_helmholtz_u(un,ub, mu=demo_nv.mu, rho=demo_nv.rho, dt=demo_nv.dt,
                                            Nx=demo_nv.Nx, Ny=demo_nv.Ny, max_iters=10000)
        
        residual_norm_l2 = np.sum(residual_u_**2)*demo_nv.dx*demo_nv.dy

        v_, residual_v_ = solve_helmholtz_v(vn,vb, mu=demo_nv.mu, rho=demo_nv.rho, dt=demo_nv.dt,
                                            Nx=demo_nv.Nx, Ny=demo_nv.Ny, max_iters=10000)

        # residual_norm_l2 = np.sum(residual_v_**2)*demo_nv.dx*demo_nv.dy

        # # 计算poisson方程的右端项
        # pb = make_pressure_b(u_, v_, Nx=demo_nv.Nx, Ny=demo_nv.Ny,
        #                     dx=demo_nv.dx, dy=demo_nv.dy, dt=demo_nv.dt, rho=demo_nv.rho)

        # p, residual_p_ = solve_poisson_p(
        #     pn,pb, Nx=demo_nv.Nx, Ny=demo_nv.Ny, max_iters=10000)
        # residual_norm_l2 = np.sum(residual_p_**2)*demo_nv.dx*demo_nv.dy

        # correct_u(u, u_, demo_nv.Nx, demo_nv.Ny, p, demo_nv.dt,
        #         demo_nv.rho, demo_nv.dx, demo_nv.dy)
        # correct_v(v, v_, demo_nv.Nx, demo_nv.Ny, p, demo_nv.dt,
        #         demo_nv.rho, demo_nv.dx, demo_nv.dy)
        
        un = np.copy(u_)
        vn = np.copy(v_)
        # pn = np.copy(p)
        
        if t > T -0.000000001:
            break


    from localtools import plot_p
    plot_p(u, demo_nv.Nx, demo_nv.Ny, title= "u")
    plot_p(v, demo_nv.Nx, demo_nv.Ny, title= "v")

    for j in range(Ny+2):
        i = demo_nv.Nx//2
        print("velocity:   ", u_[i][j])

    from localtools import plot_p
    plot_p(u, demo_nv.Nx, demo_nv.Ny, title= "u")
    plot_p(v, demo_nv.Nx, demo_nv.Ny, title= "v")



test_momentum(0.005, 128, 128)

# EE_list = []
# tt_list = []
# RR_list = []
# dt = 0.1
# demo_nv = NavierStokesDemo()
# for i in range(4, 15):
#     Nx = 16+4*i
#     Ny = 16+4*i
#     E_list = []
#     t_list = []
#     # 固定网格步长，计算不同时间步长的误差
#     for j in range(3,7):
#         dt = T/(1<<j)
#         E = test_momentum(dt, Nx, Ny)
#         E_list.append(E)
#         t_list.append(dt)
    
#     R_list = convergence_rates(t_list, E_list)
#     EE_list.append(E_list)
#     tt_list.append(t_list)
#     RR_list.append(R_list)
#     print(R_list)



# plot_convergence_rate(tt_list,EE_list,RR_list)


