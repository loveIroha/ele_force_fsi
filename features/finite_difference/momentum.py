from rhs import correct_u, correct_v
from rhs import make_pressure_b
from localtools import NEUMANN, DIRICHLET
from rhs import make_tentitive_v_b
from rhs import make_tentitive_u_b
import sympy as sym
import numpy as np


NS_EPSILON = 1e-15

class NavierStokesDemo:

    x, y, t = sym.symbols('x y t')

    dt = 0.2
    Nx = 64
    Ny = 64

    width = 1.0
    height = 1.0

    dx = width  / Nx
    dy = height / Ny

    def __init__(self,rho = 1.0, mu = 0.01):
        self.rho = rho
        self.mu = mu

        # TODO: 问题参数和求解参数
        # 构造真实解
        x, y, t = self.x, self.y, self.t
        # u = - sym.exp(t)*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256
        # v = sym.exp(-t/100)*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)/256

        # demo 1
        u = sym.exp(-t/100)*sym.sin(np.pi*x)*sym.sin(np.pi*y)
        v = sym.exp(-t/100)*sym.sin(np.pi*x)*sym.sin(np.pi*y)
        
        u = sym.exp(t)*sym.sin(np.pi*x)**2*sym.sin(2*np.pi*y)
        v = -sym.exp(t)*sym.sin(2*np.pi*x)*sym.sin(np.pi*y)**2

        # 右端项
        (self.u, self.v), (self.du, self.dv), (self.f1,
                                               self.f2) = self.make_examples_NS_2D(u, v)

    # 1. 右端项 f1 f2
    # 2. 真实解 u v p
    # 3. 梯度 （u_x u_y, v_x v_y)
    def make_examples_NS_2D(self, u, v):
        mu=self.mu
        rho=self.rho
        x, y, t = self.x, self.y, self.t
        # 生成右端项 f1 和 f2
        f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2))
        f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2))
        u_ = sym.lambdify([x, y, t], u, 'numpy')
        v_ = sym.lambdify([x, y, t], v, 'numpy')
        f1_ = sym.lambdify([x, y, t], f1, 'numpy')
        f2_ = sym.lambdify([x, y, t], f2, 'numpy')
        dudx = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
        dudy = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
        dvdx = sym.lambdify([x, y], sym.diff(v, x, 1), 'numpy')
        dvdy = sym.lambdify([x, y], sym.diff(v, y, 1), 'numpy')
        return (u_, v_), ((dudx, dudy), (dvdx, dvdy)), (f1_, f2_)

    # 返回 u, v, f1, f2 等的向量
    def array_f1(self, t):
        Nx, Ny = self.Nx, self.Ny
        f1_array = np.zeros((Nx+1, Ny+2))
        for i in range(Nx+1):
            for j in range(Ny+2):
                f1_array[i, j] = self.f1(i*self.dx, (j-0.5)*self.dy, t)

        return f1_array

    def array_u(self, t):
        Nx, Ny = self.Nx, self.Ny
        u_array = np.zeros((Nx+1, Ny+2))
        for i in range(Nx+1):
            for j in range(Ny+2):
                u_array[i, j] = self.u(i*self.dx, (j-0.5)*self.dy, t)

        return u_array

    def array_f2(self, t):
        Nx, Ny = self.Nx, self.Ny
        f2_array = np.zeros((Nx+2, Ny+1))
        for i in range(Nx+2):
            for j in range(Ny+1):
                f2_array[i, j] = self.f2((i-0.5)*self.dx, j*self.dy, t)

        return f2_array

    def array_v(self, t):
        Nx, Ny = self.Nx, self.Ny
        v_array = np.zeros((Nx+2, Ny+1))
        for i in range(Nx+2):
            for j in range(Ny+1):
                v_array[i, j] = self.v((i-0.5)*self.dx, j*self.dy, t)

        return v_array

# 输入：右端项 ub, 初解 un,
#      问题参数 rho, mu,
#      离散参数 Nx, Ny, dt
# 返回数值解 u_


def solve_helmholtz_u(un,ub,
                      mu=1.0, rho=1.0, width=1.0, height=1.0,
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

        if (np.sum(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.sum(residual**2)*dx*dy)
            break

    return phi, residual


def solve_helmholtz_v(vn,vb,
                      mu=1.0, rho=1.0, width=1.0, height=1.0,
                      all_boundary=(DIRICHLET, DIRICHLET,
                                    DIRICHLET, DIRICHLET),
                      Nx=32, Ny=32, dt=100, max_iters=10000):

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

        if (np.sum(residual**2)*dx*dy < NS_EPSILON):
            print(iter, np.sum(residual**2)*dx*dy)
            break

    return phi, residual


def test_momentum(dt, Nx, Ny):

    demo_nv.dt = dt
    demo_nv.Nx = Nx
    demo_nv.Ny = Ny
    demo_nv.dx = demo_nv.width/Nx
    demo_nv.dy = demo_nv.height/Ny

    t = 0
    un = demo_nv.array_u(0)
    vn = demo_nv.array_v(0)

    for i in range(100000000):

        t += demo_nv.dt

        f1n = demo_nv.array_f1(t)
        f2n = demo_nv.array_f2(t)

        ub = make_tentitive_u_b(un, f1n, Nx=demo_nv.Nx,
                                Ny=demo_nv.Ny, dt=demo_nv.dt, rho=demo_nv.rho)
        vb = make_tentitive_v_b(vn, f2n, Nx=demo_nv.Nx,
                                Ny=demo_nv.Ny, dt=demo_nv.dt, rho=demo_nv.rho)

        u_, residual_u_ = solve_helmholtz_u(un,ub, mu=demo_nv.mu, rho=demo_nv.rho, dt=demo_nv.dt,
                                            Nx=demo_nv.Nx, Ny=demo_nv.Ny, max_iters=10000)


        v_, residual_v_ = solve_helmholtz_v(vn,vb, mu=demo_nv.mu, rho=demo_nv.rho, dt=demo_nv.dt,
                                            Nx=demo_nv.Nx, Ny=demo_nv.Ny, max_iters=10000)


        print("time : ", t)
        print("Error of u", np.sum((u_ - demo_nv.array_u(t))**2)*demo_nv.dx*demo_nv.dy)
        print("Error of v", np.sum((v_ - demo_nv.array_v(t))**2)*demo_nv.dx*demo_nv.dy)
        # # print("Error of p",np.sum((p - demo_nv.array_p(t))**2)*demo_nv.dx*demo_nv.dy)

        un = np.copy(u_)
        vn = np.copy(v_)

        if t > 1.0-0.000000001:
            break

    error_u = np.sqrt(np.sum((u_ - demo_nv.array_u(t))**2)*demo_nv.dx*demo_nv.dy)
    error_v = np.sqrt(np.sum((v_ - demo_nv.array_v(t))**2)*demo_nv.dx*demo_nv.dy)

    print("time : ", t)
    print("Error of u", np.sum((u_ - demo_nv.array_u(t))**2)*demo_nv.dx*demo_nv.dy)
    print("Error of v", np.sum((v_ - demo_nv.array_v(t))**2)*demo_nv.dx*demo_nv.dy)

    return error_u




from localtools import convergence_rates
from localtools import plot_convergence_rate

EE_list = []
tt_list = []
RR_list = []
dt = 0.1
demo_nv = NavierStokesDemo()
for i in range(4, 7):
    Nx = 8+4*i
    Ny = 8+4*i
    E_list = []
    t_list = []
    # 固定网格步长，计算不同时间步长的误差
    for j in range(1,10):
        dt = 1.0/(1<<j)
        E = test_momentum(dt, Nx, Ny)
        E_list.append(E)
        t_list.append(dt)
    
    R_list = convergence_rates(t_list, E_list)
    EE_list.append(E_list)
    tt_list.append(t_list)
    RR_list.append(R_list)
    print(E_list)



plot_convergence_rate(tt_list,EE_list,RR_list)