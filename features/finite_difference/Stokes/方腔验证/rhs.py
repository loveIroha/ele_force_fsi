
import numpy as np


# NOTE: In fact, there is no neccessary to assign values on the bottom and top edge
def make_tentitive_u_b(un, f, Nx, Ny, dt, rho):
    ub = np.zeros((Nx+1, Ny+2))
    for i in range(0, Nx+1):
        for j in range(0, Ny+2):
            ub[i][j] = un[i][j]/dt+f[i][j]/rho

    return ub


# NOTE: In fact, there is no neccessary to assign values on the left and right edge
def make_tentitive_v_b(vn, f, Nx, Ny, dt, rho):
    vb = np.zeros((Nx+2, Ny+1))
    for i in range(0, Nx+2):
        for j in range(0, Ny+1):
            vb[i][j] = vn[i][j]/dt+f[i][j]/rho

    return vb


def make_pressure_b(u, v, Nx, Ny, dx, dy, dt, rho):
    p = np.zeros((Nx+2, Ny+2))
    for i in range(0, Nx):
        for j in range(0, Ny):
            p[i+1][j+1] = rho/dt*((u[i+1][j+1] - u[i][j+1]) / dx 
                                + (v[i+1][j+1] - v[i+1][j]) / dy)

    return p


def test_divergence_free(u, v, Nx, Ny, dx, dy, dt, rho):
    for i in range(0, Nx):
        for j in range(0, Ny):
            print((u[i+1][j+1] - u[i][j+1]) / dx 
                                + (v[i+1][j+1] - v[i+1][j]) / dy)


def correct_u(u, u_, Nx, Ny, p, dt, rho, dx, dy):
    for i in range(0, Nx+1):
    # for i in range(1, Nx):
        for j in range(1, Ny+1):
            u[i][j] = u_[i][j] - dt/rho/dx*(p[i+1][j] - p[i][j])


def correct_v(v, v_, Nx, Ny, p, dt, rho, dx, dy):
    for i in range(1, Nx+1):
        # for j in range(1, Ny):
        for j in range(0, Ny+1):
            v[i][j] = v_[i][j] - dt/rho/dy*(p[i][j+1] - p[i][j])
