import sympy as sym

file_name = "heat_equations_6.json"


class NavierStokesDemo:
    x, y, z, t = sym.symbols("x y z t")
    ccode = {}

    def __init__(
        self, Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nt = Nt

        self.width = width
        self.height = height
        self.depth = depth
        self.T = T

        self.dx = width / Nx
        self.dy = height / Ny
        self.dz = depth / Nz
        self.dt = T / Nt

        self.rho = rho
        self.mu = mu

        # 构造真实解
        # 1.
        x, y, z, t = self.x, self.y, self.z, self.t
        from sympy import sin, cos, exp
        pi = 3.1415926535897932
        u_s = pi*pi * sin(pi*x)*sin(pi*x)*sin(2.0*pi*y) * \
            sin(2.0*pi*z)*sym.exp(-t)
        v_s = -0.5*pi*pi * sin(2.0*pi*x)*sin(pi*y) * \
            sin(pi*y)*sin(2.0*pi*z)*sym.exp(-t)
        w_s = -0.5*pi*pi * sin(2.0*pi*x)*sin(2.0*pi*y) * \
            sin(pi*z)*sin(pi*z)*sym.exp(-t)
        p_s = cos(pi*x)*cos(pi*y)*cos(pi*z)
        assert sym.simplify(sym.diff(u_s, x, 1) +
                            sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1)) == 0

        # # 2. Channel flow （Poiseuille flow)
        # u_s = 4*y*(1-y)
        # v_s = 0
        # w_s = 0
        # # p_s = 8*(1-x)
        # p_s = cos(pi*x)*cos(pi*y)*cos(pi*z)

        # # 3. Beltrami flow
        # a = pi/4
        # d = pi/2
        # nv = 1
        # u_s = -a*(exp(a*x)*sin(a*y+d*z)+exp(a*z)*cos(a*x+d*y))*exp(-d*d*t)
        # v_s = -a*(exp(a*y)*sin(a*z+d*x)+exp(a*x)*cos(a*y+d*z))*exp(-d*d*t)
        # w_s = -a*(exp(a*z)*sin(a*x+d*y)+exp(a*y)*cos(a*z+d*x))*exp(-d*d*t)
        # p_s = -a*a*exp(-2*d*d*t)*(exp(2*a*x)+exp(2*a*y)+exp(2*a*z))*(sin(a*x+d*y)*cos(a*z+d*x)*exp(
        #     a*(y+z))+sin(a*y+d*z)*cos(a*x+d*y)*exp(a*(x+z))+sin(a*z+d*x)*cos(a*y+d*z)*exp(a*(x+y)))
        # assert sym.simplify(sym.diff(u_s, x, 1) +
        #                     sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1)) == 0

        # # 4. (3D Taylor-Green vortex flow ?)
        # u_s = sin(x)*cos(y)*cos(z)*exp(-t)
        # v_s = cos(x)*sin(y)*cos(z)*exp(-t)
        # w_s = -2*cos(x)*cos(y)*sin(z)*exp(-t)
        # p_s = cos(pi*x)*cos(pi*y)*cos(pi*z)
        # assert sym.simplify(sym.diff(u_s, x, 1) +
        #                     sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1)) == 0

        # # 5.
        # u_s = (y*y*y*y+z*z) * sym.exp(-t)
        # v_s = (z*z*z*z+x*x) * sym.exp(-t)
        # w_s = (x*x*x*x+y*y) * sym.exp(-t)
        # p_s = (2 * x - 1) * (2 * y - 1) * (2 * z - 1) * sym.exp(-t)
        # assert sym.simplify(sym.diff(u_s, x, 1) +
        #                     sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1)) == 0

        # 右端项
        (
            (self.u, self.v, self.w, self.p),
            (self.du, self.dv, self.dw, self.dp),
            (self.f1, self.f2, self.f3),
        ) = self.make_examples_NS_3D(u_s, v_s, w_s, p_s)

        # 临时测试项 \Delta p
        self.bp = sym.diff(p_s, x, 2) + sym.diff(p_s,
                                                 y, 2) + sym.diff(p_s, z, 2)
        self.ccode["bp"] = sym.printing.ccode(self.bp)
        # print(sym.simplify(self.bp))
        self.bp = sym.lambdify([x, y, z, t], self.bp, "numpy")

        # 测试无散度条件
        div_u = sym.simplify(sym.diff(u_s, x, 1) +
                             sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1))
        print(sym.simplify(div_u))

    def make_examples_NS_3D(self, u, v, w, p):
        mu = self.mu
        rho = self.rho
        x, y, z, t = self.x, self.y, self.z, self.t
        Nx, Ny, Nz, Nt = self.Nx, self.Ny, self.Nz, self.Nt
        width, height, depth, T = self.width, self.height, self.depth, self.T
        # 1. 右端项 f1 f2 f3
        f1 = rho * sym.diff(u, t, 1) - mu * (
            sym.diff(u, x, 2) + sym.diff(u, y, 2) + sym.diff(u, z, 2)
        )  # + sym.diff(p, x, 1)
        f2 = rho * sym.diff(v, t, 1) - mu * (
            sym.diff(v, x, 2) + sym.diff(v, y, 2) + sym.diff(v, z, 2)
        )  # + sym.diff(p, y, 1)
        f3 = rho * sym.diff(w, t, 1) - mu * (
            sym.diff(w, x, 2) + sym.diff(w, y, 2) + sym.diff(w, z, 2)
        )  # + sym.diff(p, z, 1)

        self.ccode["u"] = sym.printing.ccode(u)
        self.ccode["v"] = sym.printing.ccode(v)
        self.ccode["w"] = sym.printing.ccode(w)
        self.ccode["p"] = sym.printing.ccode(p)

        self.ccode["f1"] = sym.printing.ccode(f1)
        self.ccode["f2"] = sym.printing.ccode(f2)
        self.ccode["f3"] = sym.printing.ccode(f3)

        self.ccode["dudx"] = sym.printing.ccode(sym.diff(u, x, 1))
        self.ccode["dudy"] = sym.printing.ccode(sym.diff(u, y, 1))
        self.ccode["dudz"] = sym.printing.ccode(sym.diff(u, z, 1))
        self.ccode["dvdx"] = sym.printing.ccode(sym.diff(v, x, 1))
        self.ccode["dvdy"] = sym.printing.ccode(sym.diff(v, y, 1))
        self.ccode["dvdz"] = sym.printing.ccode(sym.diff(v, z, 1))
        self.ccode["dwdx"] = sym.printing.ccode(sym.diff(w, x, 1))
        self.ccode["dwdy"] = sym.printing.ccode(sym.diff(w, y, 1))
        self.ccode["dwdz"] = sym.printing.ccode(sym.diff(w, z, 1))
        self.ccode["dpdx"] = sym.printing.ccode(sym.diff(p, x, 1))
        self.ccode["dpdy"] = sym.printing.ccode(sym.diff(p, y, 1))
        self.ccode["dpdz"] = sym.printing.ccode(sym.diff(p, z, 1))

        self.ccode["Nx"] = Nx
        self.ccode["Ny"] = Ny
        self.ccode["Nz"] = Nz
        self.ccode["Nt"] = Nt
        self.ccode["width"] = width
        self.ccode["height"] = height
        self.ccode["depth"] = depth
        self.ccode["T"] = T
        self.ccode["rho"] = rho
        self.ccode["mu"] = mu

        f1_ = sym.lambdify([x, y, z, t], f1, "numpy")
        f2_ = sym.lambdify([x, y, z, t], f2, "numpy")
        f3_ = sym.lambdify([x, y, z, t], f3, "numpy")
        # 2. 真实解 u v w p
        u_ = sym.lambdify([x, y, z, t], u, "numpy")
        v_ = sym.lambdify([x, y, z, t], v, "numpy")
        w_ = sym.lambdify([x, y, z, t], w, "numpy")
        p_ = sym.lambdify([x, y, z, t], p, "numpy")
        # 3. 梯度 （u_x u_y u_z, v_x v_y v_z, w_x w_y w_z)
        dudx = sym.lambdify([x, y, z, t], sym.diff(u, x, 1), "numpy")
        dudy = sym.lambdify([x, y, z, t], sym.diff(u, y, 1), "numpy")
        dudz = sym.lambdify([x, y, z, t], sym.diff(u, z, 1), "numpy")
        dvdx = sym.lambdify([x, y, z, t], sym.diff(v, x, 1), "numpy")
        dvdy = sym.lambdify([x, y, z, t], sym.diff(v, y, 1), "numpy")
        dvdz = sym.lambdify([x, y, z, t], sym.diff(v, z, 1), "numpy")
        dwdx = sym.lambdify([x, y, z, t], sym.diff(w, x, 1), "numpy")
        dwdy = sym.lambdify([x, y, z, t], sym.diff(w, y, 1), "numpy")
        dwdz = sym.lambdify([x, y, z, t], sym.diff(w, z, 1), "numpy")
        dpdx = sym.lambdify([x, y, z, t], sym.diff(p, x, 1), "numpy")
        dpdy = sym.lambdify([x, y, z, t], sym.diff(p, y, 1), "numpy")
        dpdz = sym.lambdify([x, y, z, t], sym.diff(p, z, 1), "numpy")
        return (
            (u_, v_, w_, p_),
            (
                (dudx, dudy, dudz),
                (dvdx, dvdy, dvdz),
                (dwdx, dwdy, dwdz),
                (dpdx, dpdy, dpdz),
            ),
            (f1_, f2_, f3_),
        )


if __name__ == "__main__":
    N = 10
    Nx, Ny, Nz, Nt = N, N, N, 4
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    rho, mu = 1.0, 1.0
    t = 0.0
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu
    )

    import json
    with open(file_name, 'w') as json_file:
        json.dump(ns_demo.ccode, json_file, indent=4)
