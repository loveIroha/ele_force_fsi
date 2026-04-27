import numpy as np

from poisson import NEUMANN, DIRICHLET
from poisson import get_boundary_type
from poisson import generate_u_prime
from poisson import residual_prime,residual

from IO3D import write_vector_3D

from MG import MG
from generate_demo import NavierStokesDemo


def fun(N):
    Nx, Ny, Nz, Nt = N,N,N, 4
    width, height, depth, T = 1.0, 1.0, 1.0, 1.0
    rho, mu = 1.0, 0.04
    t= 0.0
    all_boundary_type = [ NEUMANN, NEUMANN, NEUMANN, NEUMANN, NEUMANN,NEUMANN]
    # all_boundary_type = [DIRICHLET, DIRICHLET, DIRICHLET, DIRICHLET,DIRICHLET,  DIRICHLET]
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
    r_prime = residual_prime(p_prime,bp,Nx, Ny, Nz, width, height, depth)

    write_vector_3D(boundary_types,'boundary_types.txt')
    write_vector_3D(boundary_values, 'boundary_values')
    write_vector_3D(p_prime, 'p_prime.txt')
    write_vector_3D(bp,'bp.txt')
    write_vector_3D(p_exact,'p_exact.txt')
    write_vector_3D(r_prime,'r_prime.txt')

    mg_solver = MG(Nx, Ny,Nz, width, height,depth, boundary_types)
    
    # mg_solver.multi_r[0] = r_prime
    # mg_solver.test_restrict()

    # write_vector_3D(mg_solver.multi_bc[4],'mg_bc_4.txt')
    

    # print(boundary_type)
    max_iters = 10000
    ph = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for iter in range(max_iters):
        ph = mg_solver.iterate(ph, r_prime, 0)
        r = residual(ph,r_prime,boundary_types, Nx, Ny, Nz, width, height, depth)
        print(iter, np.sqrt(np.sum(r**2)*dx*dy*dz))
        ph = ph - sum(sum(sum(ph)))/Nx/Ny/Nz
        if (np.sqrt(np.sum(r**2)*dx*dy*dz) < 1e-7):
            break
    
    # write_vector_3D(mg_solver.multi_e[0],'mg_e_0.txt')
    # write_vector_3D(mg_solver.multi_e[1],'mg_e_1.txt')
    # write_vector_3D(mg_solver.multi_e[2],'mg_e_2.txt')
    # write_vector_3D(mg_solver.multi_e[3],'mg_e_3.txt')    
    
    # write_vector_3D(mg_solver.multi_b[0],'mg_b_0.txt')
    # write_vector_3D(mg_solver.multi_b[1],'mg_b_1.txt')
    # write_vector_3D(mg_solver.multi_b[2],'mg_b_2.txt')
    # write_vector_3D(mg_solver.multi_b[3],'mg_b_3.txt')
    
    # write_vector_3D(mg_solver.multi_x[0],'mg_x_0.txt')
    # write_vector_3D(mg_solver.multi_x[1],'mg_x_1.txt')
    # write_vector_3D(mg_solver.multi_x[2],'mg_x_2.txt')
    # write_vector_3D(mg_solver.multi_x[3],'mg_x_3.txt')
    
    # write_vector_3D(mg_solver.multi_r[0],'mg_r_0.txt')
    # write_vector_3D(mg_solver.multi_r[1],'mg_r_1.txt')
    # write_vector_3D(mg_solver.multi_r[2],'mg_r_2.txt')
    # write_vector_3D(mg_solver.multi_r[3],'mg_r_3.txt')
    print(sum(sum(sum(ph)))/Nx/Ny/Nz)
    write_vector_3D(ph,'ph.txt')
    write_vector_3D(ph+p_prime,'p.txt')
    e = np.zeros((Nx + 2, Ny + 2, Nz + 2))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                e[i+1,j+1,k+1] = ph[i+1,j+1,k+1]+p_prime[i+1,j+1,k+1]-p_exact[i+1,j+1,k+1]
    # e = ph+p_prime - p_exact
    print(np.sqrt(np.sum(e**2)*dx*dy*dz))


if __name__ == "__main__":
    # fun(4)
    # fun(8)
    # fun(16)
    fun(32)
