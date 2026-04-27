from a import NavierStokesDemo
import numpy as np

from IO3D import write_vector_3D

NEUMANN = 2
DIRICHLET = 1


def make_tentitive_u_b(un, f, Nx, Ny, Nz, dt, rho):
    ub = np.zeros((Nx+1, Ny+2, Nz+2))
    for i in range(0, Nx+1):
        for j in range(0, Ny+2):
            for k in range(0, Nz+2):
                ub[i, j, k] = rho*un[i, j, k]/dt+f[i, j, k]

    return ub


def generate_u_prime(boundary_type, boundary_values, bu, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    u = np.zeros((Nx + 1, Ny + 2, Nz + 2))

    for i in range(0, Nx+1):
        for k in range(0, Nz):
            if boundary_type[i, 0, k + 1] == DIRICHLET:
                u[i, 0, k + 1] = 2 * boundary_values[i, 0, k + 1] - u[i, 1, k + 1]
            if boundary_type[i, 0, k + 1] == NEUMANN:
                u[i, 0, k + 1] = u[i, 1, k + 1] - \
                    dy * boundary_values[i, 0, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == DIRICHLET:
                u[i, Ny + 1, k + 1] = 2 * boundary_values[i,
                                                          Ny + 1, k + 1] - u[i, Ny, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == NEUMANN:
                u[i, Ny + 1, k + 1] = u[i, Ny, k + 1] + \
                    dy * boundary_values[i, Ny + 1, k + 1]

    # 左右
    for i in range(0, Nx+1):
        for j in range(0, Ny):
            if boundary_type[i, j + 1, 0] == DIRICHLET:
                u[i, j + 1, 0] = 2 * \
                    boundary_values[i, j + 1, 0] - u[i, j + 1, 1]
            if boundary_type[i, j + 1, 0] == NEUMANN:
                u[i, j + 1, 0] = u[i, j + 1, 1] - \
                    dz * boundary_values[i, j + 1, 0]
            if boundary_type[i, j + 1, Nz + 1] == DIRICHLET:
                u[i, j + 1, Nz + 1] = 2 * boundary_values[i,
                                                          j + 1, Nz + 1] - u[i, j + 1, Nz]
            if boundary_type[i, j + 1, Nz + 1] == NEUMANN:
                u[i, j + 1, Nz + 1] = u[i, j + 1, Nz] + \
                    dz * boundary_values[i, j + 1, Nz + 1]
    for j in range(0, Ny):
        for k in range(0, Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                u[0, j + 1, k + 1] = boundary_values[0, j+1, k+1]
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                # u_ghost 为 u[-1, j+1, k+1]
                u_ghost = u[1, j + 1, k + 1]-2.0 * \
                    dx*boundary_values[0, j + 1, k + 1]
                b1 = (u_ghost + u[1,     j + 1, k + 1]) / dx/dx
                b2 = (u[0,     j,     k + 1] +
                      u[0,     j + 2, k + 1]) / dy/dy
                b3 = (u[0,     j + 1, k] +
                      u[0,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                u[0, j+1, k+1] = (bu[0, j+1, k+1]+b)/a
            if boundary_type[Nx, j + 1, k + 1] == DIRICHLET:
                u[Nx, j + 1, k + 1] = boundary_values[Nx, j+1, k+1]
            if boundary_type[Nx, j + 1, k + 1] == NEUMANN:
                u_ghost = u[Nx-1, j + 1, k + 1]+2.0 * \
                    dx*boundary_values[Nx, j + 1, k + 1]
                b1 = (u[Nx - 1, j + 1, k + 1] + u_ghost) / dx/dx
                b2 = (u[Nx,     j,     k + 1] +
                      u[Nx,     j + 2, k + 1]) / dy/dy
                b3 = (u[Nx,     j + 1, k] +
                      u[Nx,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                u[Nx, j+1, k+1] = (bu[Nx, j+1, k+1]+b)/a



    return u


def smooth(u, bu, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    for j in range(0, Ny):
        for k in range(0, Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                u[0, j + 1, k + 1] = 0
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                u_ghost = u[1, j + 1, k + 1]
                b1 = (u_ghost + u[1,     j + 1, k + 1]) / dx/dx
                b2 = (u[0,     j,     k + 1] + u[0,     j + 2, k + 1]) / dy/dy
                b3 = (u[0,     j + 1, k] + u[0,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                u[0, j+1, k+1] = (bu[0, j+1, k+1]+b)/a
            if boundary_type[Nx, j + 1, k + 1] == DIRICHLET:
                u[Nx, j + 1, k + 1] = 0
            if boundary_type[Nx, j + 1, k + 1] == NEUMANN:
                u_ghost = u[Nx-1, j + 1, k + 1]
                b1 = (u[Nx - 1, j + 1, k + 1] + u_ghost) / dx/dx
                b2 = (u[Nx,     j,     k + 1] +
                      u[Nx,     j + 2, k + 1]) / dy/dy
                b3 = (u[Nx,     j + 1, k] + u[Nx,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                u[Nx, j+1, k+1] = (bu[Nx, j+1, k+1]+b)/a

    for i in range(0, Nx+1):
        for k in range(0, Nz):
            if boundary_type[i, 0, k + 1] == DIRICHLET:
                u[i, 0, k + 1] = - u[i, 1, k + 1]
            if boundary_type[i, 0, k + 1] == NEUMANN:
                u[i, 0, k + 1] = u[i, 1, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == DIRICHLET:
                u[i, Ny + 1, k + 1] = - u[i, Ny, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == NEUMANN:
                u[i, Ny + 1, k + 1] = u[i, Ny, k + 1]

    for i in range(0, Nx+1):
        for j in range(0, Ny):
            if boundary_type[i, j + 1, 0] == DIRICHLET:
                u[i, j + 1, 0] = - u[i, j + 1, 1]
            if boundary_type[i, j + 1, 0] == NEUMANN:
                u[i, j + 1, 0] = u[i, j + 1, 1]
            if boundary_type[i, j + 1, Nz + 1] == DIRICHLET:
                u[i, j + 1, Nz + 1] = - u[i, j + 1, Nz]
            if boundary_type[i, j + 1, Nz + 1] == NEUMANN:
                u[i, j + 1, Nz + 1] = u[i, j + 1, Nz]

    for i in range(1, Nx):
        for j in range(0, Ny):
            for k in range(0, Nz):
                b1 = (u[i - 1, j + 1, k + 1] + u[i + 1, j + 1, k + 1]) / dx/dx
                b2 = (u[i,     j,     k + 1] + u[i,     j + 2, k + 1]) / dy/dy
                b3 = (u[i,     j + 1, k] + u[i,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                u[i, j+1, k+1] = (bu[i, j+1, k+1]+b)/a
    return u


def residual_prime(u, bu, boundary_type, boundary_values, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+1, Nx+2, Ny+2))
    for i in range(0, Nx+1):
        for k in range(0, Nz):
            if boundary_type[i, 0, k + 1] == DIRICHLET:
                u[i, 0, k + 1] = 2 * boundary_values[i, 0, k + 1] - u[i, 1, k + 1]
            if boundary_type[i, 0, k + 1] == NEUMANN:
                u[i, 0, k + 1] = u[i, 1, k + 1] - \
                    dy * boundary_values[i, 0, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == DIRICHLET:
                u[i, Ny + 1, k + 1] = 2 * boundary_values[i,
                                                          Ny + 1, k + 1] - u[i, Ny, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == NEUMANN:
                u[i, Ny + 1, k + 1] = u[i, Ny, k + 1] + \
                    dy * boundary_values[i, Ny + 1, k + 1]

    for i in range(0, Nx+1):
        for j in range(0, Ny):
            if boundary_type[i, j + 1, 0] == DIRICHLET:
                u[i, j + 1, 0] = 2 * boundary_values[i, j + 1, 0] - u[i, j + 1, 1]
            if boundary_type[i, j + 1, 0] == NEUMANN:
                u[i, j + 1, 0] = u[i, j + 1, 1] - \
                    dz * boundary_values[i, j + 1, 0]
            if boundary_type[i, j + 1, Nz + 1] == DIRICHLET:
                u[i, j + 1, Nz + 1] = 2 * boundary_values[i,
                                                          j + 1, Nz + 1] - u[i, j + 1, Nz]
            if boundary_type[i, j + 1, Nz + 1] == NEUMANN:
                u[i, j + 1, Nz + 1] = u[i, j + 1, Nz] + \
                    dz * boundary_values[i, j + 1, Nz + 1]

    for j in range(0, Ny):
        for k in range(0, Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                u[0, j + 1, k + 1] = boundary_values[0, j + 1, k + 1]
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                u_ghost = u[1, j + 1, k + 1]-2.0*dx * \
                    boundary_values[0, j + 1, k + 1]
                b1 = (u_ghost + u[1,     j + 1, k + 1]) / dx/dx
                b2 = (u[0,     j,     k + 1] + u[0,     j + 2, k + 1]) / dy/dy
                b3 = (u[0,     j + 1, k] + u[0,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[0, j+1, k+1] = bu[0, j+1, k+1]+b-a*u[0, j+1, k+1]
                # u[0,j+1,k+1]=(bu[0,j+1,k+1]+b)/a
            if boundary_type[Nx, j + 1, k + 1] == DIRICHLET:
                u[Nx, j + 1, k + 1] = boundary_values[Nx, j + 1, k + 1]
            if boundary_type[Nx, j + 1, k + 1] == NEUMANN:
                u_ghost = u[Nx-1, j + 1, k + 1]+2.0 * \
                    dx*boundary_values[Nx, j + 1, k + 1]
                # u_ghost = u[Nx-1, j + 1, k + 1]
                b1 = (u[Nx - 1, j + 1, k + 1] + u_ghost) / dx/dx
                b2 = (u[Nx,     j,     k + 1] +
                      u[Nx,     j + 2, k + 1]) / dy/dy
                b3 = (u[Nx,     j + 1, k] + u[Nx,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[Nx, j+1, k+1] = bu[Nx, j+1, k+1]+b-a*u[Nx, j+1, k+1]
                # u[Nx,j+1,k+1]=(bu[Nx,j+1,k+1]+b)/a

    for i in range(1, Nx):
        for j in range(0, Ny):
            for k in range(0, Nz):
                b1 = (u[i - 1, j + 1, k + 1] + u[i + 1, j + 1, k + 1]) / dx/dx
                b2 = (u[i,     j,     k + 1] + u[i,     j + 2, k + 1]) / dy/dy
                b3 = (u[i,     j + 1, k] + u[i,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i, j+1, k+1] = bu[i, j+1, k+1]+b-a*u[i, j+1, k+1]
    return r


def residual(u, bu, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+1, Nx+2, Ny+2))
    for j in range(0, Ny):
        for k in range(0, Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                u[0, j + 1, k + 1] = 0
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                u_ghost = u[1, j + 1, k + 1]
                b1 = (u_ghost + u[1,     j + 1, k + 1]) / dx/dx
                b2 = (u[0,     j,     k + 1] + u[0,     j + 2, k + 1]) / dy/dy
                b3 = (u[0,     j + 1, k] + u[0,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[0, j+1, k+1] = bu[0, j+1, k+1]+b-a*u[0, j+1, k+1]
                # u[0,j+1,k+1]=(bu[0,j+1,k+1]+b)/a
            if boundary_type[Nx, j + 1, k + 1] == DIRICHLET:
                u[Nx, j + 1, k + 1] = 0
            if boundary_type[Nx, j + 1, k + 1] == NEUMANN:
                u_ghost = u[Nx-1, j + 1, k + 1]
                b1 = (u[Nx - 1, j + 1, k + 1] + u_ghost) / dx/dx
                b2 = (u[Nx,     j,     k + 1] +
                      u[Nx,     j + 2, k + 1]) / dy/dy
                b3 = (u[Nx,     j + 1, k] + u[Nx,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[Nx, j+1, k+1] = bu[Nx, j+1, k+1]+b-a*u[Nx, j+1, k+1]
                # u[Nx,j+1,k+1]=(bu[Nx,j+1,k+1]+b)/a

    for i in range(0, Nx+1):
        for k in range(0, Nz):
            if boundary_type[i, 0, k + 1] == DIRICHLET:
                u[i, 0, k + 1] = - u[i, 1, k + 1]
            if boundary_type[i, 0, k + 1] == NEUMANN:
                u[i, 0, k + 1] = u[i, 1, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == DIRICHLET:
                u[i, Ny + 1, k + 1] = - u[i, Ny, k + 1]
            if boundary_type[i, Ny + 1, k + 1] == NEUMANN:
                u[i, Ny + 1, k + 1] = u[i, Ny, k + 1]

    for i in range(0, Nx+1):
        for j in range(0, Ny):
            if boundary_type[i, j + 1, 0] == DIRICHLET:
                u[i, j + 1, 0] = - u[i, j + 1, 1]
            if boundary_type[i, j + 1, 0] == NEUMANN:
                u[i, j + 1, 0] = u[i, j + 1, 1]
            if boundary_type[i, j + 1, Nz + 1] == DIRICHLET:
                u[i, j + 1, Nz + 1] = - u[i, j + 1, Nz]
            if boundary_type[i, j + 1, Nz + 1] == NEUMANN:
                u[i, j + 1, Nz + 1] = u[i, j + 1, Nz]

    for i in range(1, Nx):
        for j in range(0, Ny):
            for k in range(0, Nz):
                b1 = (u[i - 1, j + 1, k + 1] + u[i + 1, j + 1, k + 1]) / dx/dx
                b2 = (u[i,     j,     k + 1] + u[i,     j + 2, k + 1]) / dy/dy
                b3 = (u[i,     j + 1, k] + u[i,     j + 1, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i, j+1, k+1] = bu[i, j+1, k+1]+b-a*u[i, j+1, k+1]
    return r


def get_boundary_type(Nx, Ny, Nz, all_boundary_type):
    # 左右下上前后
    boundary_types = np.zeros((Nx + 1, Ny + 2, Nz + 2))
    for j in range(0, Ny):
        for k in range(0, Nz):
            boundary_types[0, j + 1, k + 1] = all_boundary_type[0]
            boundary_types[Nx, j + 1, k + 1] = all_boundary_type[1]
    for i in range(0, Nx+1):
        for k in range(0, Nz):
            boundary_types[i, 0, k + 1] = all_boundary_type[2]
            boundary_types[i, Ny + 1, k + 1] = all_boundary_type[3]
    for i in range(0, Nx+1):
        for j in range(0, Ny):
            boundary_types[i, j + 1, 0] = all_boundary_type[4]
            boundary_types[i, j + 1, Nz + 1] = all_boundary_type[5]

    return boundary_types


def fun(N, Nt):
    Nx, Ny, Nz, Nt = N, N, N, Nt
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    all_boundary_type = [NEUMANN, NEUMANN,
                         NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET]
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    rho, mu, t = 1.0, 1.0, 0.0
    max_iters = 1
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
    )

    boundary_types = get_boundary_type(Nx, Ny, Nz, all_boundary_type)

    un = ns_demo.get_array_u(0.0)
    write_vector_3D(un, 'u0.txt')
    write_vector_3D(boundary_types, 'ubt.txt')
    for i in range(1, Nt+1):
        t = i*dt
        # print("NO. ", i, ", time: ", t)
        f1 = ns_demo.get_array_f1(t)
        b_u = make_tentitive_u_b(un, f1, Nx, Ny, Nz, dt, rho)
        boundary_values = ns_demo.get_boundary_values_u(
            boundary_types, ns_demo.u, ns_demo.du, t
        )
        u_prime = generate_u_prime(
            boundary_types, boundary_values, b_u, Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        u_exact = ns_demo.get_array_u(t)
        bu_hat = residual_prime(u_prime, b_u, boundary_types, boundary_values,
                                Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        uh = np.zeros((Nx + 1, Ny + 2, Nz + 2))
        for iter in range(max_iters):
            uh = smooth(uh, bu_hat, boundary_types, Nx, Ny,
                        Nz, width, height, depth, mu, rho, dt)
            r = residual(uh, bu_hat, boundary_types, Nx, Ny,
                         Nz, width, height, depth, mu, rho, dt)
            r_norm2 = np.sqrt(np.sum(r**2)*dx*dy*dz)
            print(iter, r_norm2)
            if (r_norm2 < 1e-7):
                # print(iter,r_norm2)
                break
        write_vector_3D(uh, 'uh.txt')
        write_vector_3D(r, 'r.txt')

        e = np.zeros((Nx + 1, Ny + 2, Nz + 2))
        for i in range(1, Nx):
            for j in range(0, Ny):
                for k in range(0, Nz):
                    e[i, j+1, k+1] = uh[i, j+1, k+1] + \
                        u_prime[i, j+1, k+1]-u_exact[i, j+1, k+1]

        un = np.copy(uh+u_prime)

    print("最终误差：", np.sqrt(np.sum(e**2)*dx*dy*dz))

    write_vector_3D(u_exact, 'u_exact.txt')
    write_vector_3D(un, 'un.txt')
    write_vector_3D(f1, 'f1.txt')
    write_vector_3D(boundary_types, 'boundary_types.txt')
    write_vector_3D(boundary_values, 'boundary_values.txt')
    write_vector_3D(u_prime, 'u_prime.txt')
    write_vector_3D(b_u, 'b_u.txt')
    write_vector_3D(bu_hat, 'bu_hat.txt')

    # write_vector_3D(r,'r.txt')
    # write_vector_3D(uh,'uh.txt')
    # write_vector_3D(uh+u_prime,'u.txt')
    # write_vector_3D(e,'e.txt')


if __name__ == "__main__":
    fun(4, 1)
    # for Ns in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    #     print("循环")
    #     for Nt in [1, 2, 4, 8, 16, 32, 64]:
    #         fun(Ns, Nt)
