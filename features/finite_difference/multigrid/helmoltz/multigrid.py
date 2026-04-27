from solve_demo import generate_u_prime, generate_u_residual
from solve_demo import generate_v_prime, generate_v_residual
from solve_demo import NS_EPSILON
from solve_demo import make_tentitive_u_b, make_tentitive_v_b, solve_helmholtz_u, solve_helmholtz_v
from write_demo import NavierStokesDemo
import numpy as np
from localtools import plot_convergence_rate
from localtools import convergence_rates
from localtools import NEUMANN, DIRICHLET

from mg_u import MG as MG_u
from mg_v import MG as MG_v


def solve_helmholtz_u_mg(un, ub, mu, rho, width, height, boundary_type, boundary_values, Nx, Ny, dt, max_iters, u_exact):
    dx = width/Nx
    dy = height/Ny
    phi = np.copy(un)
    residual = np.zeros((Nx+1, Ny+2))
    error = np.zeros((Nx+1, Ny+2))
    # 生成 u' 和 ub'
    phi_prime = generate_u_prime(
        boundary_type, boundary_values, ub, Nx, Ny, width, height, dt, rho, mu)
    ub = generate_u_residual(phi_prime, ub,
                             boundary_type, boundary_values, Nx, Ny, width, height, dt, rho, mu)
    # 定义多重网格求解器
    mg_solver = MG_u(Nx, Ny, width, height, dt, rho, mu, boundary_type)
    for _ in range(20):
        phi = mg_solver.iterate(phi, ub, 0)
        error_norm_l2 = np.sqrt(np.sum(mg_solver.multi_r[0]**2)*dx*dy)
        print(f"residual norm l2: {error_norm_l2}")

    phi = phi + phi_prime
    return phi, residual

def solve_helmholtz_v_mg(vn, vb, mu, rho, width, height, boundary_type, boundary_values, Nx, Ny, dt, max_iters, v_exact):
    dx = width/Nx
    dy = height/Ny
    phi = np.copy(vn)
    residual = np.zeros((Nx+2, Ny+1))
    error = np.zeros((Nx+2, Ny+1))
    # 生成 u' 和 ub'
    phi_prime = generate_v_prime(
        boundary_type, boundary_values, vb, Nx, Ny, width, height, dt, rho, mu)
    vb = generate_v_residual(phi_prime, vb,
                             boundary_type, boundary_values, Nx, Ny, width, height, dt, rho, mu)
    # 定义多重网格求解器
    mg_solver = MG_v(Nx, Ny, width, height, dt, rho, mu, boundary_type)
    for _ in range(20):
        phi = mg_solver.iterate(phi, vb, 0)
        error_norm_l2 = np.sqrt(np.sum(mg_solver.multi_r[0]**2)*dx*dy)
        print(f"residual norm l2: {error_norm_l2}")

    phi = phi + phi_prime
    return phi, residual

def solve_momentum(Nx=24, Ny=24, Nt=100, width=1.0, height=1.0, T=1.0, rho=1.0, mu=1.0):
    all_boundary_type = [NEUMANN, NEUMANN, NEUMANN, NEUMANN]
    demo_nv = NavierStokesDemo(
        Nx, Ny, Nt, width, height, T, rho, mu, all_boundary_type)
    uns, vns, f1ns, f2ns, ubvns, vbvns, btu, btv = demo_nv.generate_data()

    Nx = demo_nv.Nx
    Ny = demo_nv.Ny
    Nt = demo_nv.Nt

    dx = demo_nv.dx
    dy = demo_nv.dy
    dt = demo_nv.dt

    width = demo_nv.width
    height = demo_nv.height
    T = demo_nv.T

    rho = demo_nv.rho
    mu = demo_nv.mu

    un = uns[0]
    vn = vns[0]
    for i in range(1, Nt+1):
        t = i*dt
        print("NO. ", i, ", time: ", t)
        ub = make_tentitive_u_b(un, f1ns[i], Nx, Ny, dt, rho)
        vb = make_tentitive_v_b(vn, f2ns[i], Nx, Ny, dt, rho)

        u_, residual_u_ = solve_helmholtz_u_mg(
            un, ub, mu, rho, width, height, btu, ubvns[i], Nx, Ny, dt, 10000, uns[i])
        # u_, residual_u_ = solve_helmholtz_u(
        #     un, ub, mu, rho, width, height, btu, ubvns[i], Nx, Ny, dt, 10000)

        v_, residual_v_ = solve_helmholtz_v_mg(
            vn, vb, mu, rho, width, height, btv, vbvns[i], Nx, Ny, dt, 10000, vns[i])
        # v_, residual_v_ = solve_helmholtz_v(
        #     vn, vb, mu, rho, width, height, btv, vbvns[i], Nx, Ny, dt, 10000)
        un = np.copy(u_)
        vn = np.copy(v_)

    error_u = np.sqrt(np.sum((u_ - uns[Nt])**2)*demo_nv.dx*demo_nv.dy)
    error_v = np.sqrt(np.sum((v_ - vns[Nt])**2)*demo_nv.dx*demo_nv.dy)

    print(error_u, error_v)
    return error_v


if __name__ == "__main__":
    EE_list = []
    tt_list = []
    RR_list = []
    width_2 = 1.0
    height_2 = 1.0
    T_2 = 1.0

    rho_2 = 1.0
    mu_2 = 0.01
    for i in range(10, 20):
        Nx = 8+4*i
        Ny = 8+4*i
        E_list = []
        t_list = []
        # 固定网格步长，计算不同时间步长的误差
        for j in range(1, 5):
            Nt = 1 << j
            print("Nx Ny Nt : ", Nx, Ny, Nt)
            E = solve_momentum(Nx, Ny, Nt, width=width_2,
                               height=height_2, T=T_2, rho=rho_2, mu=mu_2)
            E_list.append(E)
            t_list.append(T_2/Nt)

        R_list = convergence_rates(t_list, E_list)
        EE_list.append(E_list)
        tt_list.append(t_list)
        RR_list.append(R_list)
        print(E_list)

    plot_convergence_rate(tt_list, EE_list, RR_list)
