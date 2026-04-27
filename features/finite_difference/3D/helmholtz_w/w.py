from a import NavierStokesDemo
import numpy as np

from IO3D import write_vector_3D

NEUMANN = 2
DIRICHLET = 1


def make_tentitive_w_b(wn, f, Nx, Ny, Nz, dt, rho):
    wb = np.zeros((Nx+2, Ny+2, Nz+1))
    for i in range(0, Nx+2):
        for j in range(0, Ny+2):
            for k in range(0, Nz+1):
                wb[i, j, k] = rho*wn[i, j, k]/dt+f[i, j, k]

    return wb


def generate_w_prime(boundary_type, boundary_values, bw, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    w = np.zeros((Nx + 2, Ny + 2, Nz + 1))
    for j in range(0, Ny):
        for k in range(0, Nz+1):
            if boundary_type[0, j + 1, k] == DIRICHLET:
                w[0, j + 1, k] = -w[1, j + 1, k] + \
                    2.0*boundary_values[0, j + 1, k]
            if boundary_type[0, j + 1, k] == NEUMANN:
                w[0, j + 1, k] = w[1, j + 1, k] - \
                    dx*boundary_values[0, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == DIRICHLET:
                w[Nx+1, j + 1, k] = -w[Nx, j + 1, k] + \
                    2.0*boundary_values[Nx+1, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == NEUMANN:
                w[Nx+1, j + 1, k] = w[Nx, j + 1, k] + \
                    dx*boundary_values[Nx+1, j + 1, k]

    for i in range(0, Nx):
        for k in range(0, Nz+1):
            if boundary_type[i+1, 0, k] == DIRICHLET:
                w[i+1, 0, k] = - w[i+1, 1, k] + 2.0*boundary_values[i+1, 0, k]
            if boundary_type[i+1, 0, k] == NEUMANN:
                w[i+1, 0, k] = w[i+1, 1, k] - dy*boundary_values[i+1, 0, k]
            if boundary_type[i+1, Ny + 1, k] == DIRICHLET:
                w[i+1, Ny + 1, k] = - w[i+1, Ny, k] + \
                    2.0*boundary_values[i+1, Ny + 1, k]
            if boundary_type[i+1, Ny + 1, k] == NEUMANN:
                w[i+1, Ny + 1, k] = w[i+1, Ny, k] + \
                    dy*boundary_values[i+1, Ny + 1, k]

    for i in range(0, Nx):
        for j in range(0, Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                w[i + 1, j + 1, 0] = boundary_values[i + 1, j + 1, 0]
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                w_ghost = w[i + 1, j + 1, 1] - 2.0 * \
                    dz*boundary_values[i + 1, j + 1, 0]
                b1 = (w[i,     j + 1, 0] + w[i + 2, j + 1, 0]) / dx/dx
                b2 = (w[i + 1, j,     0] + w[i + 1, j + 2, 0]) / dy/dy
                b3 = (w_ghost + w[i + 1, j + 1, 1]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                w[i+1, j+1, 0] = (bw[i+1, j+1, 0]+b)/a
            if boundary_type[i + 1, j + 1, Nz] == DIRICHLET:
                w[i + 1, j + 1, Nz] = boundary_values[i + 1, j + 1, Nz]
            if boundary_type[i + 1, j + 1, Nz] == NEUMANN:
                w_ghost = w[i + 1, j + 1, Nz-1] + 2.0 * \
                    dz*boundary_values[i + 1, j + 1, Nz]
                b1 = (w[i,     j + 1, Nz] + w[i + 2, j + 1, Nz]) / dx/dx
                b2 = (w[i + 1, j,     Nz] + w[i + 1, j + 2, Nz]) / dy/dy
                b3 = (w[i + 1, j + 1, Nz - 1] + w_ghost) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                w[i+1, j+1, Nz] = (bw[i+1, j+1, Nz]+b)/a

    return w


def smooth_w(w, bw, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz

    for i in range(0, Nx):
        for j in range(0, Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                w[i + 1, j + 1, 0] = 0
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                w_ghost = w[i + 1, j + 1, 1]
                b1 = (w[i,     j + 1, 0] + w[i + 2, j + 1, 0]) / dx/dx
                b2 = (w[i + 1, j,     0] + w[i + 1, j + 2, 0]) / dy/dy
                b3 = (w_ghost + w[i + 1, j + 1, 1]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                w[i+1, j+1, 0] = (bw[i+1, j+1, 0]+b)/a
            if boundary_type[i + 1, j + 1, Nz] == DIRICHLET:
                w[i + 1, j + 1, Nz] = 0
            if boundary_type[i + 1, j + 1, Nz] == NEUMANN:
                w_ghost = w[i + 1, j + 1, Nz-1]
                b1 = (w[i,     j + 1, Nz] + w[i + 2, j + 1, Nz]) / dx/dx
                b2 = (w[i + 1, j,     Nz] + w[i + 1, j + 2, Nz]) / dy/dy
                b3 = (w[i + 1, j + 1, Nz - 1] + w_ghost) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                w[i+1, j+1, Nz] = (bw[i+1, j+1, Nz]+b)/a
    for j in range(0, Ny):
        for k in range(0, Nz+1):
            if boundary_type[0, j + 1, k] == DIRICHLET:
                w[0, j + 1, k] = -w[1, j + 1, k]
            if boundary_type[0, j + 1, k] == NEUMANN:
                w[0, j + 1, k] = w[1, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == DIRICHLET:
                w[Nx+1, j + 1, k] = -w[Nx, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == NEUMANN:
                w[Nx+1, j + 1, k] = w[Nx, j + 1, k]

    for i in range(0, Nx):
        for k in range(0, Nz+1):
            if boundary_type[i+1, 0, k] == DIRICHLET:
                w[i+1, 0, k] = - w[i+1, 1, k]
            if boundary_type[i+1, 0, k] == NEUMANN:
                w[i+1, 0, k] = w[i+1, 1, k]
            if boundary_type[i+1, Ny + 1, k] == DIRICHLET:
                w[i+1, Ny + 1, k] = - w[i+1, Ny, k]
            if boundary_type[i+1, Ny + 1, k] == NEUMANN:
                w[i+1, Ny + 1, k] = w[i+1, Ny, k]

    for i in range(0, Nx):
        for j in range(0, Ny):
            for k in range(1, Nz):
                b1 = (w[i,     j + 1, k] + w[i + 2, j + 1, k]) / dx/dx
                b2 = (w[i + 1, j,     k] + w[i + 1, j + 2, k]) / dy/dy
                b3 = (w[i + 1, j + 1, k - 1] + w[i + 1, j + 1, k + 1]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                w[i+1, j+1, k] = (bw[i+1, j+1, k]+b)/a
    return w


def residual_prime(w, bw, boundary_type, boundary_values, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+2, Nx+2, Ny+1))
    for j in range(0, Ny):
        for k in range(0, Nz+1):
            if boundary_type[0, j + 1, k] == DIRICHLET:
                w[0, j + 1, k] = -w[1, j + 1, k] + \
                    2.0*boundary_values[0, j + 1, k]
                # w[0, j + 1, k] = 2.0*boundary_values[0, j + 1, k]
            if boundary_type[0, j + 1, k] == NEUMANN:
                w[0, j + 1, k] = w[1, j + 1, k] - \
                    dx*boundary_values[0, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == DIRICHLET:
                w[Nx+1, j + 1, k] = -w[Nx, j + 1, k] + \
                    2.0*boundary_values[Nx+1, j + 1, k]
                # w[Nx+1, j + 1, k] = 2.0*boundary_values[Nx+1, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == NEUMANN:
                w[Nx+1, j + 1, k] = w[Nx, j + 1, k] + \
                    dx*boundary_values[Nx+1, j + 1, k]

    for i in range(0, Nx):
        for k in range(0, Nz+1):
            if boundary_type[i+1, 0, k] == DIRICHLET:
                w[i+1, 0, k] = - w[i+1, 1, k] + 2.0*boundary_values[i+1, 0, k]
                # w[i+1, 0, k] = 2.0*boundary_values[i+1, 0, k]
            if boundary_type[i+1, 0, k] == NEUMANN:
                w[i+1, 0, k] = w[i+1, 1, k] - dy*boundary_values[i+1, 0, k]
            if boundary_type[i+1, Ny + 1, k] == DIRICHLET:
                w[i+1, Ny + 1, k] = - w[i+1, Ny, k] + \
                    2.0*boundary_values[i+1, Ny + 1, k]
                # w[i+1, Ny + 1, k] = 2.0*boundary_values[i+1, Ny + 1, k]
            if boundary_type[i+1, Ny + 1, k] == NEUMANN:
                w[i+1, Ny + 1, k] = w[i+1, Ny, k] + \
                    dy*boundary_values[i+1, Ny + 1, k]

    for i in range(0, Nx):
        for j in range(0, Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                w[i + 1, j + 1, 0] = boundary_values[i + 1, j + 1, 0]
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                w_ghost = w[i + 1, j + 1, 1] - 2.0 * \
                    dz*boundary_values[i + 1, j + 1, 0]
                b1 = (w[i,     j + 1, 0] - 2.0 *
                      w[i + 1, j + 1, 0] + w[i + 2, j + 1, 0]) / dx/dx
                b2 = (w[i + 1, j, 0] - 2.0 * w[i + 1, j + 1, 0] +
                      w[i + 1, j + 2, 0]) / dy/dy
                b3 = (w_ghost - 2.0 * w[i + 1, j + 1,
                      0] + w[i + 1, j + 1, 1]) / dz/dz
                r[i+1, j+1, 0] = bw[i+1, j+1, 0] - \
                    rho/dt*w[i+1, j+1, 0]+mu*(b1+b2+b3)
            if boundary_type[i + 1, j + 1, Nz] == DIRICHLET:
                w[i + 1, j + 1, Nz] = boundary_values[i + 1, j + 1, Nz]
            if boundary_type[i + 1, j + 1, Nz] == NEUMANN:
                w_ghost = w[i + 1, j + 1, Nz-1] + 2.0 * \
                    dz*boundary_values[i + 1, j + 1, Nz]
                b1 = (w[i,     j + 1, Nz] - 2.0 *
                      w[i + 1, j + 1, Nz] + w[i + 2, j + 1, Nz]) / dx/dx
                b2 = (w[i + 1, j, Nz] - 2.0 * w[i + 1, j + 1, Nz] +
                      w[i + 1, j + 2, Nz]) / dy/dy
                b3 = (w[i + 1, j + 1, Nz - 1] - 2.0 *
                      w[i + 1, j + 1, Nz] + w_ghost) / dz/dz
                r[i+1, j+1, Nz] = bw[i+1, j+1, Nz] - \
                    rho/dt*w[i+1, j+1, Nz]+mu*(b1+b2+b3)

    for i in range(0, Nx):
        for j in range(0, Ny):
            for k in range(1, Nz):
                b1 = (w[i,     j + 1, k] - 2.0 *
                      w[i + 1, j + 1, k] + w[i + 2, j + 1, k]) / dx/dx
                b2 = (w[i + 1, j, k] - 2.0 * w[i + 1, j + 1, k] +
                      w[i + 1, j + 2, k]) / dy/dy
                b3 = (w[i + 1, j + 1, k - 1] - 2.0 *
                      w[i + 1, j + 1, k] + w[i + 1, j + 1, k + 1]) / dz/dz
                r[i+1, j+1, k] = bw[i+1, j+1, k] - \
                    rho/dt*w[i+1, j+1, k]+mu*(b1+b2+b3)

    return r


def residual(w, bw, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+2, Nx+2, Ny+1))

    for i in range(0, Nx):
        for j in range(0, Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                w[i + 1, j + 1, 0] = 0
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                w_ghost = w[i + 1, j + 1, 1]
                b1 = (w[i,     j + 1, 0] - 2.0 *
                      w[i + 1, j + 1, 0] + w[i + 2, j + 1, 0]) / dx/dx
                b2 = (w[i + 1, j, 0] - 2.0 * w[i + 1, j + 1, 0] +
                      w[i + 1, j + 2, 0]) / dy/dy
                b3 = (w_ghost - 2.0 * w[i + 1, j + 1,
                      0] + w[i + 1, j + 1, 1]) / dz/dz
                r[i+1, j+1, 0] = bw[i+1, j+1, 0] - \
                    rho/dt*w[i+1, j+1, 0]+mu*(b1+b2+b3)
            if boundary_type[i + 1, j + 1, Nz] == DIRICHLET:
                w[i + 1, j + 1, Nz] = 0
            if boundary_type[i + 1, j + 1, Nz] == NEUMANN:
                w_ghost = w[i + 1, j + 1, Nz-1]
                b1 = (w[i,     j + 1, Nz] - 2.0 *
                      w[i + 1, j + 1, Nz] + w[i + 2, j + 1, Nz]) / dx/dx
                b2 = (w[i + 1, j, Nz] - 2.0 * w[i + 1, j + 1, Nz] +
                      w[i + 1, j + 2, Nz]) / dy/dy
                b3 = (w[i + 1, j + 1, Nz - 1] - 2.0 *
                      w[i + 1, j + 1, Nz] + w_ghost) / dz/dz
                r[i+1, j+1, Nz] = bw[i+1, j+1, Nz] - \
                    rho/dt*w[i+1, j+1, Nz]+mu*(b1+b2+b3)

    for j in range(0, Ny):
        for k in range(0, Nz+1):
            if boundary_type[0, j + 1, k] == DIRICHLET:
                w[0, j + 1, k] = -w[1, j + 1, k]
            if boundary_type[0, j + 1, k] == NEUMANN:
                w[0, j + 1, k] = w[1, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == DIRICHLET:
                w[Nx+1, j + 1, k] = -w[Nx, j + 1, k]
            if boundary_type[Nx+1, j + 1, k] == NEUMANN:
                w[Nx+1, j + 1, k] = w[Nx, j + 1, k]

    for i in range(0, Nx):
        for k in range(0, Nz+1):
            if boundary_type[i+1, 0, k] == DIRICHLET:
                w[i+1, 0, k] = - w[i+1, 1, k]
            if boundary_type[i+1, 0, k] == NEUMANN:
                w[i+1, 0, k] = w[i+1, 1, k]
            if boundary_type[i+1, Ny + 1, k] == DIRICHLET:
                w[i+1, Ny + 1, k] = - w[i+1, Ny, k]
            if boundary_type[i+1, Ny + 1, k] == NEUMANN:
                w[i+1, Ny + 1, k] = w[i+1, Ny, k]

    for i in range(0, Nx):
        for j in range(0, Ny):
            for k in range(1, Nz):
                b1 = (w[i,     j + 1, k] + w[i + 2, j + 1, k]) / dx/dx
                b2 = (w[i + 1, j, k] + w[i + 1, j + 2, k]) / dy/dy
                b3 = (w[i + 1, j + 1, k - 1] + w[i + 1, j + 1, k + 1]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, j+1, k] = bw[i+1, j+1, k]+b-a*w[i+1, j+1, k]
                # b1 = (w[i,     j + 1, k] - 2.0 *
                #       w[i + 1, j + 1, k] + w[i + 2, j + 1, k]) / dx/dx
                # b2 = (w[i + 1, j, k] - 2.0 * w[i + 1, j + 1, k] +
                #       w[i + 1, j + 2, k]) / dy/dy
                # b3 = (w[i + 1, j + 1, k - 1] - 2.0 *
                #       w[i + 1, j + 1, k] + w[i + 1, j + 1, k + 1]) / dz/dz
                # r[i+1, j+1, k] = bw[i+1, j+1, k] - \
                #     rho/dt*w[i+1, j+1, k]+mu*(b1+b2+b3)
    return r


def get_boundary_type(Nx, Ny, Nz, all_boundary_type):
    # 左右下上前后
    boundary_types = np.zeros((Nx + 2, Ny + 2, Nz + 1))
    for j in range(0, Ny):
        for k in range(0, Nz+1):
            boundary_types[0,    j + 1, k] = all_boundary_type[0]
            boundary_types[Nx+1, j + 1, k] = all_boundary_type[1]
    for i in range(0, Nx):
        for k in range(0, Nz+1):
            boundary_types[i+1,  0,     k] = all_boundary_type[2]
            boundary_types[i+1, Ny + 1, k] = all_boundary_type[3]
    for i in range(0, Nx):
        for j in range(0, Ny):
            boundary_types[i+1, j + 1, 0] = all_boundary_type[4]
            boundary_types[i+1, j + 1, Nz] = all_boundary_type[5]

    return boundary_types


def fun(N, Nt):
    Nx, Ny, Nz, Nt = N, N, N, Nt
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    all_boundary_type = [NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET]
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    rho, mu, t = 1.0, 1.0, 0.0
    max_iters = 20000
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
    )

    boundary_types = get_boundary_type(Nx, Ny, Nz, all_boundary_type)

    wn = ns_demo.get_array_w(0.0)
    write_vector_3D(wn, 'w0.txt')
    write_vector_3D(boundary_types, 'wbt.txt')
    for i in range(1, Nt+1):
        t = i*dt
        # print("NO. ", i, ", time: ", t)
        f3 = ns_demo.get_array_f3(t)
        b_w = make_tentitive_w_b(wn, f3, Nx, Ny, Nz, dt, rho)
        boundary_values = ns_demo.get_boundary_values_w(
            boundary_types, ns_demo.w, ns_demo.dw, t
        )
        w_prime = generate_w_prime(
            boundary_types, boundary_values, b_w, Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        write_vector_3D(w_prime, 'w_prime.txt')
        w_exact = ns_demo.get_array_w(t)
        bw_hat = residual_prime(w_prime, b_w, boundary_types, boundary_values,
                                Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        wh = np.zeros((Nx + 2, Ny + 2, Nz + 1))
        for iter in range(max_iters):
            wh = smooth_w(wh, bw_hat, boundary_types, Nx, Ny,
                          Nz, width, height, depth, mu, rho, dt)
            r = residual(wh, bw_hat, boundary_types, Nx, Ny,
                         Nz, width, height, depth, mu, rho, dt)
            r_norm2 = np.sqrt(np.sum(r**2)*dx*dy*dz)
            # print("r_norm2: ", r_norm2)
            if (r_norm2 < 1e-7):
                # print(iter, r_norm2)
                break
            write_vector_3D(wh, 'uh.txt')
            write_vector_3D(r, 'r.txt')
            
            e = np.zeros((Nx + 2, Ny + 2, Nz + 1))
            for i in range(0, Nx):
                for j in range(0, Ny):
                    for k in range(1, Nz):
                        e[i+1, j+1, k] = wh[i+1, j+1, k] + \
                            w_prime[i+1, j+1, k]-w_exact[i+1, j+1, k]

            write_vector_3D(e, 'e.txt')

        wn = np.copy(wh+w_prime)

    print("e_norm2: ", np.sqrt(np.sum(e**2)*dx*dy*dz))

    # write_vector_3D(w_exact,'w_exact.txt')
    # write_vector_3D(wn,'wn.txt')
    # write_vector_3D(f3,'f3.txt')
    # write_vector_3D(boundary_types,'boundary_types.txt')
    # write_vector_3D(boundary_values,'boundary_values.txt')
    # write_vector_3D(b_w, 'b_w.txt')
    # write_vector_3D(bw_hat, 'bw_hat.txt')
    # write_vector_3D(r,'r.txt')
    # write_vector_3D(wh,'wh.txt')
    # write_vector_3D(wh+w_prime,'u.txt')
    # write_vector_3D(e,'e.txt')


if __name__ == "__main__":
    # fun(4, 1)
    for Ns in [4, 8, 16, 32, 64]:
        print("循环")
        for Nt in [1, 2, 4, 8, 16]:
            fun(Ns, Nt)
