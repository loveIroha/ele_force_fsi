import sympy as sym
import numpy as np
from localtools import NEUMANN, DIRICHLET


import numpy as np

# 计算收敛率
DIRICHLET = 1
NEUMANN = 2


def get_boundary_type(Nx, Ny, all_boundary_type):
    boundary_type = np.zeros((Nx+2, Ny+2))

    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary_type[0]
        boundary_type[i][Ny+1] = all_boundary_type[1]

    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary_type[2]
        boundary_type[Nx+1][j] = all_boundary_type[3]

    return boundary_type


def get_boundary_type_v(Nx, Ny, all_boundary):
    boundary_type = np.zeros((Nx+2, Ny+1))
    for i in range(1, Nx+1):
        boundary_type[i][0] = all_boundary[0]
        boundary_type[i][Ny] = all_boundary[1]

    for j in range(0, Ny+1):
        boundary_type[0][j] = all_boundary[2]
        boundary_type[Nx+1][j] = all_boundary[3]

    return boundary_type


def get_boundary_type_u(Nx, Ny, all_boundary):
    boundary_type = np.zeros((Nx+1, Ny+2))

    for i in range(0, Nx+1):
        boundary_type[i][0] = all_boundary[0]
        boundary_type[i][Ny+1] = all_boundary[1]

    for j in range(1, Ny+1):
        boundary_type[0][j] = all_boundary[2]
        boundary_type[Nx][j] = all_boundary[3]

    return boundary_type


def get_boundary_values_u(t, u, du, Nx, Ny, width, height, boundary_type):
    boundary_values = np.zeros((Nx+1, Ny+2))
    dx = width/Nx
    dy = height/Ny

    for i in range(0, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(dx*i, 0, t)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](dx*i, 0, t)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(dx*i, height, t)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](dx*i, height, t)

    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j-0.5*dy, t)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j-0.5*dy, t)
        if boundary_type[Nx][j] == DIRICHLET:
            boundary_values[Nx][j] = u(width, dy*j-0.5*dy, t)
        if boundary_type[Nx][j] == NEUMANN:
            boundary_values[Nx][j] = du[0](width, dy*j-0.5*dy, t)

    return boundary_values


def get_boundary_values_v(t, u, du, Nx, Ny, width, height, boundary_type):
    boundary_values = np.zeros((Nx+2, Ny+1))
    dx = width/Nx
    dy = height/Ny
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(dx*i-0.5*dx, 0, t)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](dx*i-0.5*dx, 0, t)
        if boundary_type[i][Ny] == DIRICHLET:
            boundary_values[i][Ny] = u(dx*i-0.5*dx, height, t)
        if boundary_type[i][Ny] == NEUMANN:
            boundary_values[i][Ny] = du[1](dx*i-0.5*dx, height, t)

    for j in range(0, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j, t)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j, t)
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(width, dy*j, t)
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](width, dy*j, t)

    return boundary_values


def get_boundary_values(t, u, du, Nx, Ny, width, height, boundary_type):
    boundary_values = np.zeros((Nx+2, Ny+2))
    dx = width/Nx
    dy = height/Ny
    for i in range(1, Nx+1):
        if boundary_type[i][0] == DIRICHLET:
            boundary_values[i][0] = u(dx*i-0.5*dx, 0, t)
        if boundary_type[i][0] == NEUMANN:
            boundary_values[i][0] = du[1](dx*i-0.5*dx, 0, t)
        if boundary_type[i][Ny+1] == DIRICHLET:
            boundary_values[i][Ny+1] = u(dx*i-0.5*dx, height, t)
        if boundary_type[i][Ny+1] == NEUMANN:
            boundary_values[i][Ny+1] = du[1](dx*i-0.5*dx, height, t)

    for j in range(1, Ny+1):
        if boundary_type[0][j] == DIRICHLET:
            boundary_values[0][j] = u(0, dy*j-0.5*dy, t)
        if boundary_type[0][j] == NEUMANN:
            boundary_values[0][j] = du[0](0, dy*j-0.5*dy, t)
        if boundary_type[Nx+1][j] == DIRICHLET:
            boundary_values[Nx+1][j] = u(width, dy*j-0.5*dy, t)
        if boundary_type[Nx+1][j] == NEUMANN:
            boundary_values[Nx+1][j] = du[0](width, dy*j-0.5*dy, t)

    return boundary_values


all_boundary_type = [NEUMANN, DIRICHLET, DIRICHLET, DIRICHLET]


def main_1():
    Nx = 2
    Ny = 2
    width = 1.0
    height = 1.0
    def du_1(x, y): return 1+x
    def du_2(x, y): return 1+y
    du = [du_1, du_2]
    def u(x, y): return -x
    boundary_type = get_boundary_type(Nx, Ny, all_boundary_type)
    boundary_value = get_boundary_values(
        u, du, Nx, Ny, width, height, boundary_type)
    print(boundary_type.T)
    print(boundary_value.T)


def main_2():
    Nx = 2
    Ny = 2
    width = 1.0
    height = 1.0
    def du_1(x, y): return 1+y
    def du_2(x, y): return 1+x
    du = [du_1, du_2]
    def u(x, y): return -x
    boundary_type = get_boundary_type_u(Nx, Ny, all_boundary_type)
    boundary_value = get_boundary_values_u(
        u, du, Nx, Ny, width, height, boundary_type)
    print(boundary_type.T)
    print(boundary_value.T)


def main_3():
    Nx = 2
    Ny = 2
    width = 1.0
    height = 1.0
    def du_1(x, y, t): return 1+y
    def du_2(x, y, t): return 1+x
    du = [du_1, du_2]
    def u(x, y, t): return 100+100*y
    boundary_type = get_boundary_type_v(Nx, Ny, all_boundary_type)
    boundary_value = get_boundary_values_v(1,
                                           u, du, Nx, Ny, width, height, boundary_type)
    print(boundary_type.T)
    print(boundary_value.T)


# print(get_boundary_type_u(2, 2, all_boundary_type).T)
# print(get_boundary_type_v(2, 2, all_boundary_type).T)


class NavierStokesDemo:

    x, y, t = sym.symbols('x y t')

    def __init__(self, Nx, Ny, Nt, width, height, T, rho, mu, all_boundary_type):
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.width = width
        self.height = height
        self.T = T
        self.dx = width/Nx
        self.dy = height/Ny
        self.dt = T/Nt
        self.rho = rho
        self.mu = mu
        self.all_boundary_type = all_boundary_type

        # TODO: 问题参数和求解参数
        # 构造真实解
        x, y, t = self.x, self.y, self.t
        # demo 1
        u = - sym.exp(t)*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256
        v = sym.exp(-t/100)*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)/256

        # demo 2
        u = sym.exp(-t/100)*sym.sin(np.pi*x)*sym.sin(np.pi*y)
        v = sym.exp(-t/100)*sym.sin(np.pi*x)*sym.sin(np.pi*y)

        # demo 3
        u = sym.exp(t)*sym.sin(np.pi*x)**2*sym.sin(2*np.pi*y)+1.0
        v = -sym.exp(t)*sym.sin(2*np.pi*x)*sym.sin(np.pi*y)**2+1.0

        # 右端项
        (self.u, self.v), (self.du, self.dv), (self.f1,
                                               self.f2) = self.make_examples_NS_2D(u, v)

    # 1. 右端项 f1 f2
    # 2. 真实解 u v p
    # 3. 梯度 （u_x u_y, v_x v_y)
    def make_examples_NS_2D(self, u, v):
        mu = self.mu
        rho = self.rho
        x, y, t = self.x, self.y, self.t
        # 生成右端项 f1 和 f2
        f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2))
        f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2))
        u_ = sym.lambdify([x, y, t], u, 'numpy')
        v_ = sym.lambdify([x, y, t], v, 'numpy')
        f1_ = sym.lambdify([x, y, t], f1, 'numpy')
        f2_ = sym.lambdify([x, y, t], f2, 'numpy')
        dudx = sym.lambdify([x, y, t], sym.diff(u, x, 1), 'numpy')
        dudy = sym.lambdify([x, y, t], sym.diff(u, y, 1), 'numpy')
        dvdx = sym.lambdify([x, y, t], sym.diff(v, x, 1), 'numpy')
        dvdy = sym.lambdify([x, y, t], sym.diff(v, y, 1), 'numpy')
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

    def save_data_single(self, u, demo_nv, variable, i):
        filename = "data/demo_1_{Nx}_{Ny}_{Nt}_{variable:s}_{i:d}.txt"
        # filename1 = "demo_1_{Nx}_{Ny}_{Nt}_{width}_{height}_{T}_{rho}_{mu}_{variable:s}_{i:d}.txt"
        # print(filename1.format(variable="u", i=i,Nx=demo_nv.Nx, Ny=demo_nv.Ny, Nt=demo_nv.Nt, width=demo_nv.width, height=demo_nv.height, T=demo_nv.T, rho=demo_nv.rho, mu=demo_nv.mu))
        f = filename.format(variable=variable, i=i,
                            Nx=demo_nv.Nx, Ny=demo_nv.Ny, Nt=demo_nv.Nt)
        np.savetxt(f, u)

    def generate_data(self):
        t = 0
        self.uns = []
        self.vns = []
        self.f1ns = []
        self.f2ns = []
        self.ubvns = []
        self.vbvns = []
        self.boundary_type_u = get_boundary_type_u(
            self.Nx, self.Ny, self.all_boundary_type)
        self.boundary_type_v = get_boundary_type_v(
            self.Nx, self.Ny, self.all_boundary_type)
        for i in range(self.Nt+1):
            t = i*self.dt
            un = self.array_u(t)
            vn = self.array_v(t)
            f1n = self.array_f1(t)
            f2n = self.array_f2(t)
            ubvn = get_boundary_values_u(
                t, self.u, self.du, self.Nx, self.Ny, self.width, self.height, self.boundary_type_u)
            vbvn = get_boundary_values_v(
                t, self.v, self.dv, self.Nx, self.Ny, self.width, self.height, self.boundary_type_v)
            self.uns.append(un)
            self.vns.append(vn)
            self.f1ns.append(f1n)
            self.f2ns.append(f2n)
            self.ubvns.append(ubvn)
            self.vbvns.append(vbvn)
            self.save_data_single(un, self, "un", i)
            self.save_data_single(vn, self, "vn", i)
            self.save_data_single(f1n, self, "f1n", i)
            self.save_data_single(f2n, self, "f2n", i)
            self.save_data_single(ubvn, self, "ubvn", i)
            self.save_data_single(vbvn, self, "vbvn", i)
        return self.uns, self.vns, self.f1ns, self.f2ns, self.ubvns, self.vbvns, self.boundary_type_u, self.boundary_type_v


if __name__ == """__main__""":
    main_3()
    demo_nv = NavierStokesDemo(
        Nx=64, Ny=64, Nt=100, width=1.0, height=1.0, T=1.0, rho=1.0, mu=1.0)
    demo_nv.generate_data()
