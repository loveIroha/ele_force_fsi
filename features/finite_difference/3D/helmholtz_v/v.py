from a import NavierStokesDemo
import numpy as np

from IO3D import write_vector_3D

NEUMANN = 2
DIRICHLET = 1


def make_tentitive_v_b(vn, f, Nx, Ny, Nz, dt, rho):
    vb = np.zeros((Nx+2, Ny+1, Nz+2))
    for i in range(0, Nx+2):
        for j in range(0, Ny+1):
            for k in range(0, Nz+2):
                vb[i, j, k] = rho*vn[i, j, k]/dt+f[i, j, k]

    return vb


def generate_u_prime(boundary_type, boundary_values, bv, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    v = np.zeros((Nx + 2, Ny + 1, Nz + 2))
    # 下上

    
    for i in range(0, Nx):
        for j in range(0, Ny+1):
            if boundary_type[i+1, j, 0] == DIRICHLET:
                v[i+1, j, 0] = 2 * boundary_values[i+1, j, 0] - v[i+1, j, 1]
            if boundary_type[i+1, j, 0] == NEUMANN:
                v[i+1, j, 0] = v[i+1, j, 1] - dz * boundary_values[i+1, j, 0]
            if boundary_type[i+1, j, Nz + 1] == DIRICHLET:
                v[i+1, j, Nz + 1] = 2 * \
                    boundary_values[i+1, j, Nz + 1] - v[i+1, j, Nz]
            if boundary_type[i+1, j, Nz + 1] == NEUMANN:
                v[i+1, j, Nz + 1] = v[i+1, j, Nz] + \
                    dz * boundary_values[i+1, j, Nz + 1]

    # 左右
    for j in range(0, Ny+1):
        for k in range(0, Nz):
            if boundary_type[0, j, k + 1] == DIRICHLET:
                v[0, j, k + 1] = 2 * boundary_values[0, j, k + 1] - v[1, j, k+1]
            if boundary_type[0, j, k + 1] == NEUMANN:
                v[0, j, k + 1] = v[1, j, k + 1] - dx * boundary_values[0, j, k + 1]
            if boundary_type[Nx + 1, j, k + 1] == DIRICHLET:
                v[Nx+1, j, k + 1] = 2 * boundary_values[Nx + 1, j, k + 1] - v[Nx, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == NEUMANN:
                v[Nx+1, j, k + 1] = v[Nx, j, k+1] + dx * boundary_values[Nx + 1, j, k + 1]

    for i in range(0, Nx):
        for k in range(0, Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                v[i + 1, 0, k + 1] = boundary_values[i + 1, 0, k + 1]
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                v_ghost = v[i + 1, 1, k + 1] - 2.0 * dy*boundary_values[i + 1, 0, k + 1]
                b1 = (v[i,     0,     k + 1] + v[i + 2, 0, k + 1]) / dx/dx
                b2 = (v_ghost + v[i + 1, 1, k + 1]) / dy/dy
                b3 = (v[i + 1, 0,     k] + v[i + 1, 0, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                v[i+1, 0, k+1] = (bv[i+1, 0, k+1]+b)/a
            if boundary_type[i + 1, Ny, k + 1] == DIRICHLET:
                v[i + 1, Ny, k + 1] = boundary_values[i + 1, Ny, k + 1]
            if boundary_type[i + 1, Ny, k + 1] == NEUMANN:
                v_ghost = v[i + 1, Ny - 1, k + 1] + 2.0 * \
                    dy*boundary_values[i + 1, Ny, k + 1]
                b1 = (v[i,     Ny,     k + 1] +
                      v[i + 2, Ny,     k + 1]) / dx/dx
                b2 = (v[i + 1, Ny - 1, k + 1] + v_ghost) / dy/dy
                b3 = (v[i + 1, Ny,     k] + v[i + 1, Ny,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                v[i+1, Ny, k+1] = (bv[i+1, Ny, k+1]+b)/a

    return v


def smooth(v, bv, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz

    for i in range(0, Nx):
        for k in range(0, Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                v[i + 1, 0, k + 1] = 0.0
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                v_ghost = v[i + 1, 1, k + 1]
                b1 = (v[i,     0,     k + 1] + v[i + 2, 0, k + 1]) / dx/dx
                b2 = (v_ghost + v[i + 1, 1, k + 1]) / dy/dy
                b3 = (v[i + 1, 0,     k] + v[i + 1, 0, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                v[i+1, 0, k+1] = (bv[i+1, 0, k+1]+b)/a
            if boundary_type[i + 1, Ny, k + 1] == DIRICHLET:
                v[i + 1, Ny, k + 1] = 0.0
            if boundary_type[i + 1, Ny, k + 1] == NEUMANN:
                v_ghost = v[i + 1, Ny - 1, k + 1]
                b1 = (v[i,     Ny,     k + 1] +
                      v[i + 2, Ny,     k + 1]) / dx/dx
                b2 = (v[i + 1, Ny - 1, k + 1] + v_ghost) / dy/dy
                b3 = (v[i + 1, Ny,     k] + v[i + 1, Ny,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                v[i+1, Ny, k+1] = (bv[i+1, Ny, k+1]+b)/a

    for j in range(0, Ny+1):
        for k in range(0, Nz):
            if boundary_type[0, j, k + 1] == DIRICHLET:
                v[0, j, k + 1] = - v[1, j, k+1]
            if boundary_type[0, j, k+1] == NEUMANN:
                v[0, j, k + 1] = v[1, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == DIRICHLET:
                v[Nx+1, j, k + 1] = - v[Nx, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == NEUMANN:
                v[Nx+1, j, k + 1] = v[Nx, j, k+1]

    for i in range(0, Nx):
        for j in range(0, Ny+1):
            if boundary_type[i+1, j, 0] == DIRICHLET:
                v[i+1, j, 0] = - v[i+1, j, 1]
            if boundary_type[i+1, j, 0] == NEUMANN:
                v[i+1, j, 0] = v[i+1, j, 1]
            if boundary_type[i+1, j, Nz + 1] == DIRICHLET:
                v[i+1, j, Nz + 1] = - v[i+1, j, Nz]
            if boundary_type[i+1, j, Nz + 1] == NEUMANN:
                v[i+1, j, Nz + 1] = v[i+1, j, Nz]

    for i in range(0, Nx):
        for j in range(1, Ny):
            for k in range(0, Nz):
                b1 = (v[i,     j,     k + 1] + v[i + 2, j,     k + 1]) / dx/dx
                b2 = (v[i + 1, j - 1, k + 1] + v[i + 1, j+1,   k + 1]) / dy/dy
                b3 = (v[i + 1, j,     k] + v[i + 1, j,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                v[i+1, j, k+1] = (bv[i+1, j, k+1]+b)/a
    return v


def residual_prime_v(v, bv, boundary_type, boundary_values, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+2, Ny+1, Ny+2))
    for j in range(0, Ny+1):
        for k in range(0, Nz):
            if boundary_type[0, j, k + 1] == DIRICHLET:
                v[0, j, k + 1] = 2 * boundary_values[0, j, k + 1] - v[1, j, k+1]
            if boundary_type[0, j, k + 1] == NEUMANN:
                v[0, j, k + 1] = v[1, j, k + 1] - \
                    dx * boundary_values[0, j, k + 1]
            if boundary_type[Nx + 1, j, k + 1] == DIRICHLET:
                v[Nx+1, j, k + 1] = 2 * \
                    boundary_values[Nx + 1, j, k + 1] - v[Nx, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == NEUMANN:
                v[Nx+1, j, k + 1] = v[Nx, j, k+1] + \
                    dx * boundary_values[Nx + 1, j, k + 1]

    for i in range(0, Nx):
        for j in range(0, Ny+1):
            if boundary_type[i+1, j, 0] == DIRICHLET:
                v[i+1, j, 0] = 2 * boundary_values[i+1, j, 0] - v[i+1, j, 1]
            if boundary_type[i+1, j, 0] == NEUMANN:
                v[i+1, j, 0] = v[i+1, j, 1] - dz * boundary_values[i+1, j, 0]
            if boundary_type[i+1, j, Nz + 1] == DIRICHLET:
                v[i+1, j, Nz + 1] = 2 * \
                    boundary_values[i+1, j, Nz + 1] - v[i+1, j, Nz]
            if boundary_type[i+1, j, Nz + 1] == NEUMANN:
                v[i+1, j, Nz + 1] = v[i+1, j, Nz] + \
                    dz * boundary_values[i+1, j, Nz + 1]

    for i in range(0, Nx):
        for k in range(0, Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                v[i + 1, 0, k + 1] = boundary_values[i + 1, 0, k + 1]
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                v_ghost = v[i + 1, 1, k + 1] - 2.0 * \
                    dy*boundary_values[i + 1, 0, k + 1]
                b1 = (v[i,     0,     k + 1] + v[i + 2, 0, k + 1]) / dx/dx
                b2 = (v_ghost + v[i + 1, 1, k + 1]) / dy/dy
                b3 = (v[i + 1, 0,     k] + v[i + 1, 0, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, 0, k+1] = bv[i+1, 0, k+1]+b-a*v[i+1, 0, k+1]
            if boundary_type[i + 1, Ny, k + 1] == DIRICHLET:
                v[i + 1, Ny, k + 1] = boundary_values[i + 1, Ny, k + 1]
            if boundary_type[i + 1, Ny, k + 1] == NEUMANN:
                v_ghost = v[i + 1, Ny - 1, k + 1]+2.0 * \
                    dy*boundary_values[i + 1, Ny, k + 1]
                b1 = (v[i,     Ny,     k + 1] +
                      v[i + 2, Ny,     k + 1]) / dx/dx
                b2 = (v[i + 1, Ny - 1, k + 1] + v_ghost) / dy/dy
                b3 = (v[i + 1, Ny,     k] + v[i + 1, Ny,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, Ny, k+1] = bv[i+1, Ny, k+1]+b-a*v[i+1, Ny, k+1]

    for i in range(0, Nx):
        for j in range(1, Ny):
            for k in range(0, Nz):
                b1 = (v[i,     j,     k + 1] + v[i + 2, j,     k + 1]) / dx/dx
                b2 = (v[i + 1, j - 1, k + 1] + v[i + 1, j + 1, k + 1]) / dy/dy
                b3 = (v[i + 1, j,     k] + v[i + 1, j,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, j, k+1] = bv[i+1, j, k+1]+b-a*v[i+1, j, k+1]
    return r


def residual(v, bv, boundary_type, Nx, Ny, Nz, width, height, depth, mu, rho, dt):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx+2, Nx+1, Ny+2))

    for i in range(0, Nx):
        for k in range(0, Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                v[i + 1, 0, k + 1] = 0.0
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                v_ghost = v[i + 1, 1, k + 1]
                b1 = (v[i,     0,     k + 1] + v[i + 2, 0, k + 1]) / dx/dx
                b2 = (v_ghost + v[i + 1, 1, k + 1]) / dy/dy
                b3 = (v[i + 1, 0,     k] + v[i + 1, 0, k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, 0, k+1] = bv[i+1, 0, k+1]+b-a*v[i+1, 0, k+1]
            if boundary_type[i + 1, Ny, k + 1] == DIRICHLET:
                v[i + 1, Ny, k + 1] = 0.0
            if boundary_type[i + 1, Ny, k + 1] == NEUMANN:
                v_ghost = v[i + 1, Ny - 1, k + 1]
                b1 = (v[i,     Ny,     k + 1] +
                      v[i + 2, Ny,     k + 1]) / dx/dx
                b2 = (v[i + 1, Ny - 1, k + 1] + v_ghost) / dy/dy
                b3 = (v[i + 1, Ny,     k] + v[i + 1, Ny,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, Ny, k+1] = bv[i+1, Ny, k+1]+b-a*v[i+1, Ny, k+1]

    for j in range(0, Ny+1):
        for k in range(0, Nz):
            if boundary_type[0, j, k + 1] == DIRICHLET:
                v[0, j, k + 1] = - v[1, j, k+1]
            if boundary_type[0, j, k + 1] == NEUMANN:
                v[0, j, k + 1] = v[1, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == DIRICHLET:
                v[Nx+1, j, k + 1] = - v[Nx, j, k+1]
            if boundary_type[Nx + 1, j, k + 1] == NEUMANN:
                v[Nx+1, j, k + 1] = v[Nx, j, k+1]

    for i in range(0, Nx):
        for j in range(0, Ny+1):
            if boundary_type[i+1, j, 0] == DIRICHLET:
                v[i+1, j, 0] = - v[i+1, j, 1]
            if boundary_type[i+1, j, 0] == NEUMANN:
                v[i+1, j, 0] = v[i+1, j, 1]
            if boundary_type[i+1, j, Nz + 1] == DIRICHLET:
                v[i+1, j, Nz + 1] = - v[i+1, j, Nz]
            if boundary_type[i+1, j, Nz + 1] == NEUMANN:
                v[i+1, j, Nz + 1] = v[i+1, j, Nz]

    for i in range(0, Nx):
        for j in range(1, Ny):
            for k in range(0, Nz):
                b1 = (v[i, j,     k + 1] + v[i + 2, j,     k + 1]) / dx/dx
                b2 = (v[i + 1, j - 1, k + 1] + v[i + 1, j + 1, k + 1]) / dy/dy
                b3 = (v[i + 1, j,     k] + v[i + 1, j,     k + 2]) / dz/dz
                b = mu*(b1+b2+b3)
                a = rho/dt+2*mu*(1/dx/dx+1/dy/dy+1/dz/dz)
                r[i+1, j, k+1] = bv[i+1, j, k+1]+b-a*v[i+1, j, k+1]
    return r


def get_boundary_type(Nx, Ny, Nz, all_boundary_type):
    # 左右下上前后
    boundary_types = np.zeros((Nx + 2, Ny + 1, Nz + 2))
    for j in range(0, Ny+1):
        for k in range(0, Nz):
            boundary_types[0,    j, k + 1] = all_boundary_type[0]
            boundary_types[Nx+1, j, k + 1] = all_boundary_type[1]
    for i in range(0, Nx):
        for k in range(0, Nz):
            boundary_types[i+1, 0,  k + 1] = all_boundary_type[2]
            boundary_types[i+1, Ny, k + 1] = all_boundary_type[3]
    for i in range(0, Nx):
        for j in range(0, Ny+1):
            boundary_types[i+1, j, 0] = all_boundary_type[4]
            boundary_types[i+1, j, Nz + 1] = all_boundary_type[5]

    return boundary_types


def fun(N, Nt):
    Nx, Ny, Nz, Nt = N, N, N, Nt
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    all_boundary_type = [NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET]
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    rho, mu, t = 1.0, 1.0, 0.0
    max_iters = 2
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
    )

    boundary_types = get_boundary_type(Nx, Ny, Nz, all_boundary_type)

    vn = ns_demo.get_array_v(0.0)
    write_vector_3D(vn,'v0.txt')
    write_vector_3D(boundary_types, 'vbt.txt')
    for i in range(1, Nt+1):
        t = i*dt
        # print("NO. ", i, ", time: ", t)
        f2 = ns_demo.get_array_f2(t)
        b_v = make_tentitive_v_b(vn, f2, Nx, Ny, Nz, dt, rho)
        boundary_values = ns_demo.get_boundary_values_v(
            boundary_types, ns_demo.v, ns_demo.dv, t
        )
        v_prime = generate_u_prime(
            boundary_types, boundary_values, b_v, Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        write_vector_3D(v_prime, 'v_prime.txt')
        v_exact = ns_demo.get_array_v(t)
        bv_hat = residual_prime_v(v_prime, b_v, boundary_types,
                                  boundary_values, Nx, Ny, Nz, width, height, depth, mu, rho, dt)
        vh = np.zeros((Nx + 2, Ny + 1, Nz + 2))
        for iter in range(max_iters):
            vh = smooth(vh, bv_hat, boundary_types, Nx, Ny,
                          Nz, width, height, depth, mu, rho, dt)
            r = residual(vh, bv_hat, boundary_types, Nx, Ny,
                           Nz, width, height, depth, mu, rho, dt)
            r_norm2 = np.sqrt(np.sum(r**2)*dx*dy*dz)
            print("r_norm2: ", r_norm2)
            if (r_norm2 < 1e-7):
                # print(iter,r_norm2)
                break
            write_vector_3D(vh, 'vh.txt')
            write_vector_3D(r, 'r.txt')

            e = np.zeros((Nx + 2, Ny + 1, Nz + 2))
            for i in range(0, Nx):
                for j in range(1, Ny):
                    for k in range(0, Nz):
                        e[i+1, j, k+1] = vh[i+1, j, k+1] + \
                            v_prime[i+1, j, k+1]-v_exact[i+1, j, k+1]

            write_vector_3D(e, 'e.txt')
            print("e_norm2: ", np.sqrt(np.sum(e**2)*dx*dy*dz))

        vn = np.copy(vh+v_prime)


    # write_vector_3D(v_exact,'v_exact.txt') 
    # write_vector_3D(vn,'vn.txt')
    # write_vector_3D(f2,'f2.txt')
    # write_vector_3D(boundary_types,'vbt.txt')
    # write_vector_3D(boundary_values,'vbv.txt')
    # write_vector_3D(b_v, 'b_v.txt')
    # write_vector_3D(bv_hat, 'bv_hat.txt')
    # write_vector_3D(r,'r.txt')
    # write_vector_3D(vh,'vh.txt')
    # write_vector_3D(vh+v_prime,'v.txt')
    # write_vector_3D(e,'e.txt')


if __name__ == "__main__":
    fun(4, 1)
    # for Ns in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    #     print("循环")
    #     for Nt in [1, 2, 4, 8, 16, 32, 64]:
    #         fun(Ns, Nt)
