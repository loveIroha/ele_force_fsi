import sympy as sym
import numpy as np

NEUMANN = 2
DIRICHLET = 1


class NavierStokesDemo:
    x, y, z, t = sym.symbols("x y z t")
    ccode = {}

    def __init__(
        self, Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
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
        self.all_boundary_type = all_boundary_type

        # 构造真实解
        x, y, z, t = self.x, self.y, self.z, self.t
        u_s = (y * y * y * y + z * z * z * z + x * x * x * x) * sym.exp(-t)
        v_s = (y * y * y * y + z * z * z * z + x * x * x * x) * sym.exp(-t)
        w_s = (y * y * y * y + z * z * z * z + x * x * x * x) * sym.exp(-t)
        p_s = (2 * x - 1) * (2 * y - 1) * (2 * z - 1) * sym.exp(-t) + 1.0
        # 散度为零
        # print(sym.diff(u_s, x, 1) + sym.diff(v_s, y, 1) + sym.diff(w_s, z, 1))

        # 右端项
        (
            (self.u, self.v, self.w, self.p),
            (self.du, self.dv, self.dw, self.dp),
            (self.f1, self.f2, self.f3),
        ) = self.make_examples_NS_3D(u_s, v_s, w_s, p_s)

        # 临时测试项
        self.bp = sym.diff(p_s, x, 2) + sym.diff(p_s,
                                                 y, 2) + sym.diff(p_s, z, 2)
        self.ccode["bp"] = sym.printing.ccode(self.bp)
        # print(sym.simplify(self.bp))
        # print(sym.simplify(p_s))
        self.bp = sym.lambdify([x, y, z, t], self.bp, "numpy")

    def make_examples_NS_3D(self, u, v, w, p):
        mu = self.mu
        rho = self.rho
        x, y, z, t = self.x, self.y, self.z, self.t
        Nx, Ny, Nz, Nt = self.Nx, self.Ny, self.Nz, self.Nt
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth, T = self.width, self.height, self.depth, self.T
        # 1. 右端项 f1 f2 f3
        f1 = rho * sym.diff(u, t, 1) - mu * (
            sym.diff(u, x, 2) + sym.diff(u, y, 2) + sym.diff(u, z, 2)
        )
        f2 = rho * sym.diff(v, t, 1) - mu * (
            sym.diff(v, x, 2) + sym.diff(v, y, 2) + sym.diff(v, z, 2)
        )
        f3 = rho * sym.diff(w, t, 1) - mu * (
            sym.diff(w, x, 2) + sym.diff(w, y, 2) + sym.diff(w, z, 2)
        )

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

    def get_array_bp(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 2))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    data[i + 1, j + 1, k + 1] = self.bp(
                        (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                    # print((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t,data[i+1,j+1,k+1])
        return data

    def get_array_u(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 1, Ny + 2, Nz + 2))
        for i in range(0, Nx+1):
            for j in range(-1, Ny+1):
                for k in range(-1, Nz+1):
                    data[i, j + 1, k +
                         1] = self.u(i*dx, (j+0.5)*dy, (k+0.5)*dz, t)

        return data

    def get_array_v(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 1, Nz + 2))
        for i in range(-1, Nx+1):
            for j in range(0, Ny+1):
                for k in range(-1, Nz+1):
                    data[i + 1, j, k +
                         1] = self.v((i+0.5)*dx, j*dy, (k+0.5)*dz, t)

        return data

    def get_array_w(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 1))
        for i in range(-1, Nx + 1):
            for j in range(-1, Ny + 1):
                for k in range(0, Nz + 1):
                    data[i + 1, j + 1, k] = self.w(
                        (i + 0.5) * dx, (j + 0.5) * dy, k * dz, t
                    )

        return data

    def get_array_f1(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 1, Ny + 2, Nz + 2))
        for i in range(0, Nx+1):
            for j in range(-1, Ny+1):
                for k in range(-1, Nz+1):
                    data[i, j+1, k+1] = self.f1(i*dx,
                                                (j+0.5)*dy, (k+0.5)*dz, t)

        return data

    def get_array_f2(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 1, Nz + 2))
        for i in range(-1, Nx+1):
            for j in range(0, Ny+1):
                for k in range(-1, Nz+1):
                    data[i+1, j, k +
                         1] = self.f2((i+0.5)*dx, j*dy, (k+0.5)*dz, t)

        return data

    def get_array_f3(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 1))
        for i in range(-1, Nx + 1):
            for j in range(-1, Ny + 1):
                for k in range(0, Nz + 1):
                    data[i + 1, j + 1, k] = self.f3(
                        (i + 0.5) * dx, (j + 0.5) * dy, k * dz, t
                    )

        return data

    def get_array_p(self, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 2))
        for i in range(-1, Nx + 1):
            for j in range(-1, Ny + 1):
                for k in range(-1, Nz + 1):
                    data[i + 1, j + 1, k + 1] = self.p(
                        (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                    # print((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t,data[i+1,j+1,k+1])
        return data

    def get_boundary_values_p(self, boundary_type, p, dp, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 2))
        for j in range(Ny):
            for k in range(Nz):
                if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                    data[0, j + 1, k +
                         1] = p(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t)
                if boundary_type[0, j + 1, k + 1] == NEUMANN:
                    data[0, j + 1, k + 1] = dp[0](
                        0.0, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx + 1, j + 1, k + 1] == DIRICHLET:
                    data[Nx + 1, j + 1, k + 1] = p(
                        width, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx + 1, j + 1, k + 1] == NEUMANN:
                    data[Nx + 1, j + 1, k + 1] = dp[0](
                        width, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )

        for i in range(Nx):
            for k in range(Nz):
                if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                    data[i + 1, 0, k + 1] = p(
                        (i + 0.5) * dx, 0.0, (k + 0.5) * dz, t)
                if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                    data[i + 1, 0, k + 1] = dp[1](
                        (i + 0.5) * dx, 0.0, (k + 0.5) * dz, t
                    )
                if boundary_type[i + 1, Ny + 1, k + 1] == DIRICHLET:
                    data[i + 1, Ny + 1, k + 1] = p(
                        (i + 0.5) * dx, height, (k + 0.5) * dz, t
                    )
                if boundary_type[i + 1, Ny + 1, k + 1] == NEUMANN:
                    data[i + 1, Ny + 1, k + 1] = dp[1](
                        (i + 0.5) * dx, height, (k + 0.5) * dz, t
                    )

        for i in range(Nx):
            for j in range(Ny):
                if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                    data[i + 1, j + 1, 0] = p((i + 0.5)
                                              * dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                    data[i + 1, j + 1,
                         0] = dp[2]((i + 0.5) * dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i + 1, j + 1, Nz + 1] == DIRICHLET:
                    data[i + 1, j + 1, Nz + 1] = p(
                        (i + 0.5) * dx, (j + 0.5) * dy, depth, t
                    )
                if boundary_type[i + 1, j + 1, Nz + 1] == NEUMANN:
                    data[i + 1, j + 1, Nz + 1] = dp[2](
                        (i + 0.5) * dx, (j + 0.5) * dy, depth, t
                    )

        return data

    def get_boundary_values_u(self, boundary_type, u, du, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 1, Ny + 2, Nz + 2))
        for j in range(0, Ny):
            for k in range(0, Nz):
                if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                    data[0, j + 1, k +
                         1] = u(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t)
                if boundary_type[0, j + 1, k + 1] == NEUMANN:
                    data[0, j + 1, k + 1] = du[0](
                        0.0, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx, j + 1, k + 1] == DIRICHLET:
                    data[Nx, j + 1, k + 1] = u(
                        width, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx, j + 1, k + 1] == NEUMANN:
                    data[Nx, j + 1, k + 1] = du[0](
                        width, (j + 0.5) * dy, (k + 0.5) * dz, t
                    )

        for i in range(0, Nx+1):
            for k in range(0, Nz):
                if boundary_type[i, 0, k + 1] == DIRICHLET:
                    data[i, 0, k + 1] = u(
                        i * dx, 0.0, (k + 0.5) * dz, t
                    )
                if boundary_type[i, 0, k + 1] == NEUMANN:
                    data[i, 0, k + 1] = du[1](
                        i * dx, 0.0, (k + 0.5) * dz, t
                    )
                if boundary_type[i, Ny + 1, k + 1] == DIRICHLET:
                    data[i, Ny + 1, k + 1] = u(
                        i*dx, height, (k + 0.5) * dz, t
                    )
                if boundary_type[i, Ny + 1, k + 1] == NEUMANN:
                    data[i, Ny + 1, k + 1] = du[1](
                        i*dx, height, (k + 0.5) * dz, t
                    )

        for i in range(0, Nx+1):
            for j in range(0, Ny):
                if boundary_type[i, j + 1, 0] == DIRICHLET:
                    data[i, j + 1, 0] = u(
                        i*dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i, j + 1, 0] == NEUMANN:
                    data[i, j + 1, 0] = du[2](
                        i*dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i, j + 1, Nz + 1] == DIRICHLET:
                    data[i, j + 1, Nz+1] = u(
                        i*dx, (j + 0.5) * dy, depth, t
                    )
                if boundary_type[i, j + 1, Nz + 1] == NEUMANN:
                    data[i, j + 1, Nz + 1] = du[2](
                        i*dx, (j + 0.5) * dy, depth, t
                    )
        return data

    def get_boundary_values_v(self, boundary_type, v, dv, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 1, Nz + 2))
        for j in range(0, Ny+1):
            for k in range(0, Nz):
                if boundary_type[0, j, k + 1] == DIRICHLET:
                    data[0, j, k + 1] = v(
                        0.0, j * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[0, j, k + 1] == NEUMANN:
                    data[0, j, k + 1] = dv[0](
                        0.0, j * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx+1, j, k + 1] == DIRICHLET:
                    data[Nx+1, j, k + 1] = v(
                        width, j * dy, (k + 0.5) * dz, t
                    )
                if boundary_type[Nx+1, j, k + 1] == NEUMANN:
                    data[Nx+1, j, k + 1] = dv[0](
                        width, j * dy, (k + 0.5) * dz, t
                    )

        for i in range(0, Nx):
            for k in range(0, Nz):
                if boundary_type[i+1, 0, k + 1] == DIRICHLET:
                    data[i+1, 0, k + 1] = v(
                        (i+0.5) * dx, 0.0, (k + 0.5) * dz, t
                    )
                if boundary_type[i+1, 0, k + 1] == NEUMANN:
                    data[i+1, 0, k + 1] = dv[1](
                        (i+0.5) * dx, 0.0, (k + 0.5) * dz, t
                    )
                if boundary_type[i+1, Ny, k + 1] == DIRICHLET:
                    data[i+1, Ny, k + 1] = v(
                        (i+0.5) * dx, height, (k + 0.5) * dz, t
                    )
                if boundary_type[i+1, Ny, k + 1] == NEUMANN:
                    data[i+1, Ny, k + 1] = dv[1](
                        (i+0.5) * dx, height, (k + 0.5) * dz, t
                    )
        for i in range(0, Nx):
            for j in range(0, Ny+1):
                if boundary_type[i+1, j, 0] == DIRICHLET:
                    data[i+1, j, 0] = v(
                        (i+0.5) * dx, j * dy, 0, t)
                if boundary_type[i+1, j, 0] == NEUMANN:
                    data[i+1, j, 0] = dv[2](
                        (i+0.5) * dx, j * dy, 0, t)
                if boundary_type[i+1, j, Nz + 1] == DIRICHLET:
                    data[i+1, j, Nz+1] = v(
                        (i+0.5) * dx, j * dy, depth, t
                    )
                if boundary_type[i+1, j, Nz + 1] == NEUMANN:
                    data[i+1, j, Nz + 1] = dv[2](
                        (i+0.5) * dx, j * dy, depth, t
                    )
        return data

    def get_boundary_values_w(self, boundary_type, w, dw, t):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        width, height, depth = self.width, self.height, self.depth
        data = np.zeros((Nx + 2, Ny + 2, Nz + 1))
        for j in range(0, Ny):
            for k in range(0, Nz + 1):
                if boundary_type[0, j + 1, k] == DIRICHLET:
                    data[0, j + 1, k] = w(0.0, (j + 0.5) * dy, k * dz, t)
                if boundary_type[0, j + 1, k] == NEUMANN:
                    data[0, j + 1, k] = dw[0](0.0, (j + 0.5) * dy, k * dz, t)
                if boundary_type[Nx + 1, j + 1, k] == DIRICHLET:
                    data[Nx + 1, j + 1, k] = w(width,
                                               (j + 0.5) * dy, k * dz, t)
                if boundary_type[Nx + 1, j + 1, k] == NEUMANN:
                    # print(boundary_type[Nx + 1, j + 1, k],dw[0](width, (j + 0.5) * dy, k * dz, t))
                    data[Nx + 1, j + 1,
                         k] = dw[0](width, (j + 0.5) * dy, k * dz, t)

        for i in range(0, Nx):
            for k in range(0, Nz + 1):
                if boundary_type[i + 1, 0, k] == DIRICHLET:
                    data[i + 1, 0, k] = w((i + 0.5) * dx, 0.0, k * dz, t)
                if boundary_type[i + 1, 0, k] == NEUMANN:
                    data[i + 1, 0, k] = dw[1]((i + 0.5) * dx, 0.0, k * dz, t)
                if boundary_type[i + 1, Ny + 1, k] == DIRICHLET:
                    data[i + 1, Ny + 1,
                         k] = w((i + 0.5) * dx, height, k * dz, t)
                if boundary_type[i + 1, Ny + 1, k] == NEUMANN:
                    data[i + 1, Ny + 1,
                         k] = dw[1]((i + 0.5) * dx, height, k * dz, t)

        for i in range(0, Nx):
            for j in range(0, Ny):
                if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                    data[i + 1, j + 1, 0] = w((i + 0.5)
                                              * dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                    # print(boundary_type[Nx + 1, j + 1, k],dw[2](width, (j + 0.5) * dy, k * dz, t))
                    data[i + 1, j + 1,
                         0] = dw[2]((i + 0.5) * dx, (j + 0.5) * dy, 0, t)
                if boundary_type[i + 1, j + 1, Nz] == DIRICHLET:
                    data[i + 1, j + 1,
                         Nz] = w((i + 0.5) * dx, (j + 0.5) * dy, depth, t)
                if boundary_type[i + 1, j + 1, Nz] == NEUMANN:
                    data[i + 1, j + 1, Nz] = dw[2]((i + 0.5) * dx, (j + 0.5) * dy, depth, t
                                                   )
        return data


if __name__ == "__main__":
    N = 10
    Nx, Ny, Nz, Nt = N, N, N, 4
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    rho, mu = 1.0, 1.0
    t = 0.0
    all_boundary_type = [NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, NEUMANN,  DIRICHLET]
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
    )

    # print(ns_demo.u_s.ccode())
    print(ns_demo.ccode)

    # 指定要保存JSON数据的文件名
    file_name = "ns_demo.json"
    import json
    # 使用json.dump()函数将字典数据写入JSON文件
    with open(file_name, 'w') as json_file:
        json.dump(ns_demo.ccode, json_file, indent=4)
