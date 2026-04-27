from generate_demo import NavierStokesDemo
import numpy as np

from IO3D import write_vector_3D

NEUMANN = 2
DIRICHLET = 1



def generate_u_prime(boundary_type, boundary_values, Nx, Ny,Nz, width, height,depth):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    data = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for j in range(Ny):
        for k in range(Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                data[0, j + 1, k + 1] = 2 * boundary_values[0, j + 1, k + 1] - data[1, j + 1, k + 1]
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                data[0, j + 1, k + 1] = data[1, j + 1, k + 1] - dx * boundary_values[0, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == DIRICHLET:
                data[Nx + 1, j + 1, k + 1] = 2 * boundary_values[Nx + 1, j + 1, k + 1] - data[Nx, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == NEUMANN:
                data[Nx + 1, j + 1, k + 1] = data[Nx, j + 1, k + 1] + dx * boundary_values[Nx + 1, j + 1, k + 1]

    for i in range(Nx):
        for k in range(Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                data[i + 1, 0, k + 1] = 2 * boundary_values[i + 1, 0, k + 1] - data[i + 1, 1, k + 1]
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                data[i + 1, 0, k + 1] = data[i + 1, 1, k + 1] - dy * boundary_values[i + 1, 0, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == DIRICHLET:
                data[i + 1, Ny + 1, k + 1] = 2 * boundary_values[i + 1, Ny + 1, k + 1] - data[i + 1, Ny, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == NEUMANN:
                data[i + 1, Ny + 1, k + 1] = data[i + 1, Ny, k + 1] + dy * boundary_values[i + 1, Ny + 1, k + 1]

    for i in range(Nx):
        for j in range(Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                data[i + 1, j + 1, 0] = 2 * boundary_values[i + 1, j + 1, 0] - data[i + 1, j + 1, 1]
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                data[i + 1, j + 1, 0] = data[i + 1, j + 1, 1] - dz * boundary_values[i + 1, j + 1, 0]
            if boundary_type[i + 1, j + 1, Nz + 1] == DIRICHLET:
                data[i + 1, j + 1, Nz + 1] = 2 * boundary_values[i + 1, j + 1, Nz + 1] - data[i + 1, j + 1, Nz]
            if boundary_type[i + 1, j + 1, Nz + 1] == NEUMANN:
                data[i + 1, j + 1, Nz + 1] = data[i + 1, j + 1, Nz] + dz * boundary_values[i + 1, j + 1, Nz + 1]
    return data

def smooth(p,bp, boundary_type, Nx, Ny, Nz, width, height, depth):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    
    for j in range(Ny):
        for k in range(Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                p[0, j + 1, k + 1] = - p[1, j + 1, k + 1]
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                p[0, j + 1, k + 1] = p[1, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == DIRICHLET:
                p[Nx + 1, j + 1, k + 1] =  - p[Nx, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == NEUMANN:
                p[Nx + 1, j + 1, k + 1] = p[Nx, j + 1, k + 1]

    for i in range(Nx):
        for k in range(Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                p[i + 1, 0, k + 1] = - p[i + 1, 1, k + 1]
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                p[i + 1, 0, k + 1] = p[i + 1, 1, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == DIRICHLET:
                p[i + 1, Ny + 1, k + 1] = - p[i + 1, Ny, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == NEUMANN:
                p[i + 1, Ny + 1, k + 1] = p[i + 1, Ny, k + 1]

    for i in range(Nx):
        for j in range(Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                p[i + 1, j + 1, 0] = - p[i + 1, j + 1, 1]
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                p[i + 1, j + 1, 0] = p[i + 1, j + 1, 1]
            if boundary_type[i + 1, j + 1, Nz + 1] == DIRICHLET:
                p[i + 1, j + 1, Nz + 1] = - p[i + 1, j + 1, Nz]
            if boundary_type[i + 1, j + 1, Nz + 1] == NEUMANN:
                p[i + 1, j + 1, Nz + 1] = p[i + 1, j + 1, Nz] 
      
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if (i+j+k)%2 == 0:
                    b1 = (p[i + 2, j + 1, k + 1] + p[i, j + 1, k + 1]) / dx/dx
                    b2 = (p[i + 1, j + 2, k + 1] + p[i + 1, j, k + 1]) / dy/dy
                    b3 = (p[i + 1, j + 1, k + 2] + p[i + 1, j + 1, k]) / dz/dz
                    b = b1+b2+b3
                    a = -2/dx/dx-2/dy/dy-2/dz/dz
                    p[i + 1, j + 1, k + 1] = (bp[i + 1, j + 1, k + 1]-b)/a
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if (i+j+k)%2 == 1:
                    b1 = (p[i + 2, j + 1, k + 1] + p[i, j + 1, k + 1]) / dx/dx
                    b2 = (p[i + 1, j + 2, k + 1] + p[i + 1, j, k + 1]) / dy/dy
                    b3 = (p[i + 1, j + 1, k + 2] + p[i + 1, j + 1, k]) / dz/dz
                    b = b1+b2+b3
                    a = -2/dx/dx-2/dy/dy-2/dz/dz
                    p[i + 1, j + 1, k + 1] = (bp[i + 1, j + 1, k + 1]-b)/a
    return p

def residual(p,bp, boundary_type, Nx, Ny, Nz, width, height, depth):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for j in range(Ny):
        for k in range(Nz):
            if boundary_type[0, j + 1, k + 1] == DIRICHLET:
                p[0, j + 1, k + 1] = - p[1, j + 1, k + 1]
            if boundary_type[0, j + 1, k + 1] == NEUMANN:
                p[0, j + 1, k + 1] = p[1, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == DIRICHLET:
                p[Nx + 1, j + 1, k + 1] =  - p[Nx, j + 1, k + 1]
            if boundary_type[Nx + 1, j + 1, k + 1] == NEUMANN:
                p[Nx + 1, j + 1, k + 1] = p[Nx, j + 1, k + 1]
            pass

    for i in range(Nx):
        for k in range(Nz):
            if boundary_type[i + 1, 0, k + 1] == DIRICHLET:
                p[i + 1, 0, k + 1] = - p[i + 1, 1, k + 1]
            if boundary_type[i + 1, 0, k + 1] == NEUMANN:
                p[i + 1, 0, k + 1] = p[i + 1, 1, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == DIRICHLET:
                p[i + 1, Ny + 1, k + 1] = - p[i + 1, Ny, k + 1]
            if boundary_type[i + 1, Ny + 1, k + 1] == NEUMANN:
                p[i + 1, Ny + 1, k + 1] = p[i + 1, Ny, k + 1]
            pass

    for i in range(Nx):
        for j in range(Ny):
            if boundary_type[i + 1, j + 1, 0] == DIRICHLET:
                p[i + 1, j + 1, 0] = - p[i + 1, j + 1, 1]
            if boundary_type[i + 1, j + 1, 0] == NEUMANN:
                p[i + 1, j + 1, 0] = p[i + 1, j + 1, 1]
            if boundary_type[i + 1, j + 1, Nz + 1] == DIRICHLET:
                p[i + 1, j + 1, Nz + 1] = - p[i + 1, j + 1, Nz]
            if boundary_type[i + 1, j + 1, Nz + 1] == NEUMANN:
                p[i + 1, j + 1, Nz + 1] = p[i + 1, j + 1, Nz] 
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                ra = (p[i + 2, j + 1, k + 1] -2*p[i+1,j+1,k+1]+ p[i, j + 1, k + 1]) / dx/dx
                rb = (p[i + 1, j + 2, k + 1] -2*p[i+1,j+1,k+1]+ p[i + 1, j, k + 1]) / dy/dy
                rc = (p[i + 1, j + 1, k + 2] -2*p[i+1,j+1,k+1]+ p[i + 1, j + 1, k]) / dz/dz
                r[i + 1, j + 1, k + 1] = bp[i + 1, j + 1, k + 1] - ra-rb-rc
    
    return r




def residual_prime(p,bp, Nx, Ny, Nz, width, height, depth):
    dx = width / Nx
    dy = height / Ny
    dz = depth / Nz
    r = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                ra = (p[i + 2, j + 1, k + 1] -2*p[i+1,j+1,k+1]+ p[i, j + 1, k + 1]) / dx/dx
                rb = (p[i + 1, j + 2, k + 1] -2*p[i+1,j+1,k+1]+ p[i + 1, j, k + 1]) / dy/dy
                rc = (p[i + 1, j + 1, k + 2] -2*p[i+1,j+1,k+1]+ p[i + 1, j + 1, k]) / dz/dz
                r[i + 1, j + 1, k + 1] = bp[i + 1, j + 1, k + 1] - ra-rb-rc
    
    return r

def get_boundary_type(Nx, Ny, Nz, all_boundary_type):
    # 左右下上前后
    boundary_types = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for j in range(Ny):
        for k in range(Nz):
            boundary_types[0, j + 1, k + 1] = all_boundary_type[0]
            boundary_types[Nx + 1, j + 1, k + 1] = all_boundary_type[1]
    for i in range(Nx):
        for k in range(Nz):
            boundary_types[i + 1, 0, k + 1] = all_boundary_type[2]
            boundary_types[i + 1, Ny + 1, k + 1] = all_boundary_type[3]
    for i in range(Nx):
        for j in range(Ny):
            boundary_types[i + 1, j + 1, 0] = all_boundary_type[4]
            boundary_types[i + 1, j + 1, Nz + 1] = all_boundary_type[5]
    
    return boundary_types
    
def fun(N, Nt):
    Nx, Ny, Nz = N,N,N
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    rho, mu = 1.0, 0.04
    t= 1.0
    all_boundary_type = [NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET]
    dx, dy, dz, dt = width / Nx, height / Ny, depth / Nz, T / Nt
    ns_demo = NavierStokesDemo(
        Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu, all_boundary_type
    )

    boundary_types = get_boundary_type(Nx, Ny, Nz, all_boundary_type)

    boundary_values = ns_demo.get_boundary_values_p(
        boundary_types, ns_demo.p, ns_demo.dp,t
    )


    p_prime = generate_u_prime(boundary_types, boundary_values, Nx, Ny,Nz, width, height,depth)
    bp = ns_demo.get_array_bp(t)
    p_exact = ns_demo.get_array_p(t)
    bp_hat = residual_prime(p_prime,bp,Nx, Ny, Nz, width, height, depth)

    write_vector_3D(boundary_types,'/home/fenics/npuheart-dev/build_3D/python/pbt.txt')
    write_vector_3D(boundary_values, '/home/fenics/npuheart-dev/build_3D/python/pbv.txt')
    write_vector_3D(p_prime, '/home/fenics/npuheart-dev/build_3D/python/p_prime.txt')
    write_vector_3D(bp,'/home/fenics/npuheart-dev/build_3D/python/bp.txt')
    write_vector_3D(p_exact,'/home/fenics/npuheart-dev/build_3D/python/p_exact.txt')
    write_vector_3D(bp_hat,'/home/fenics/npuheart-dev/build_3D/python/r_prime.txt')


    # print(boundary_type)
    max_iters = 2
    ph = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for iter in range(max_iters):
        ph = smooth(ph,bp_hat, boundary_types, Nx, Ny,Nz, width, height,depth)
        r = residual(ph,bp_hat,boundary_types, Nx, Ny, Nz, width, height, depth)
        print(iter,np.sqrt(np.sum(r**2)*dx*dy*dz))
        if (np.sqrt(np.sum(r**2)*dx*dy*dz) < 1e-7):
            break
        
    write_vector_3D(ph,'ph.txt')
    write_vector_3D(ph+p_prime,'ph_p_prime.txt')
    e = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                e[i+1,j+1,k+1] = ph[i+1,j+1,k+1]+p_prime[i+1,j+1,k+1]-p_exact[i+1,j+1,k+1]
    # e = ph+p_prime - p_exact
    print(np.sqrt(np.sum(e**2)*dx*dy*dz))


if __name__ == "__main__":
    fun(32,1)
    # for Ns in [4,8,16,32,64]:
    #     print("循环")
    #     for Nt in [1,2,4,8,16,32,64]:
    #         fun(Ns,Nt)