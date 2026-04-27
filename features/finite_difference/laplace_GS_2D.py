


import sympy as sym 
import numpy as np
import matplotlib.pyplot as plt

x,y = sym.symbols('x y') 
u_1 = sym.sin(np.pi*x)*sym.sin(np.pi*y)
u_2 = x*x + y*y + x*y + 1

def laplace(u):
    return sym.diff(u, x, 2)+sym.diff(u, y, 2)

def grad(u):
    return sym.diff(u, x, 1), sym.diff(u, y, 1)


def make_example(u):
    lambda_u =  sym.lambdify([x,y], u, 'numpy')
    lambda_grad_u_x =  sym.lambdify([x,y], sym.diff(u, x, 1), 'numpy')
    lambda_grad_u_y =  sym.lambdify([x,y], sym.diff(u, y, 1), 'numpy')
    lambda_laplace_u = sym.lambdify([x,y], laplace(u), 'numpy')
    return lambda_u, (lambda_grad_u_x, lambda_grad_u_y) , lambda_laplace_u

DIRICHLET = 1
NEUMANN= 2

def fun(Nx=32,Ny=32,max_iters=100,u_1=u_1):

    # discretizing geometry
    u,du,f = make_example(u_1)
    width = 1.0
    height = 1.0
    dx = width/Nx
    dy = height/Ny

    coordinate_x = np.linspace(-dx,width+dx,Nx+2)
    coordinate_y = np.linspace(-dy,height+dy,Ny+2)

    X,Y = np.meshgrid(coordinate_x,coordinate_y)


    b = np.zeros((Nx+2,Ny+2))
    phi = np.zeros((Nx+2,Ny+2))
    u_exact = np.zeros((Nx+2,Ny+2))
    error = np.zeros((Nx+2,Ny+2))
    boundary_type = np.zeros((Nx+2,Ny+2))
    for i in range(1,Nx+1):
        boundary_type[i][0] = DIRICHLET
        boundary_type[i][Ny+1] = DIRICHLET
    
    for j in range(1,Ny+1):
        boundary_type[0][j] = DIRICHLET
        boundary_type[Nx+1][j] = DIRICHLET
    
    boundary_values = np.zeros((Nx+2,Ny+2))


    for i in range(Nx+2):
        for j in range(Ny+2):
            u_exact[i][j] = u(X[i][j],Y[i][j])
            b[i][j] = f(X[i][j],Y[i][j])


# (phi_i+1,j + phi_i-1,j - 2 phi_i,j)/dx/dx+(phi_i,j+1 + phi_i,j+1 - 2 phi_i,j)/dy/dy = b_ij

# dy*dy*(phi_i+1,j + phi_i-1,j)+dx*dx*(phi_i,j+1 + phi_i,j+1)-(2*dy*dy+2*dx*dx)phi_i,j=b_ij*dx*dx*dy*dy
# (dy*dy*(phi_i+1,j + phi_i-1,j)+dx*dx*(phi_i,j+1 + phi_i,j+1)-b_ij)/(2*dy*dy+2*dx*dx)=phi_i,j

    for iter in range(max_iters):
        # 迭代求解
        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                phi[i][j] = ((phi[i+1][j]+phi[i-1][j])*dy*dy+(phi[i][j+1]+phi[i][j-1])*dx*dx-b[i][j]*dx*dx*dy*dy)/(2*dx*dx+2*dy*dy)


        for i in range(1,Nx+1):
            for j in range(1,Ny+1):
                error[i][j] = phi[i][j] - u_exact[i][j]

    error_norm_l2 = np.sum(error**2)*dx*dy
    return error_norm_l2



def fun2(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(3,7):
        Nx = 1<<i
        Ny = 1<<i
        h = 1.0/Nx
        E = fun(Nx,Ny,max_iters=40000)
        E_list.append(E)
        h_list.append(h)
        N_list.append(Nx)


    # 计算收敛率
    def convergence_rates(h, E):
        """
        Given a sequence of discretization parameters in the list h,
        and corresponding errors in the list E,
        compute the convergence rate of two successive (h[i], E[i])
        and (h[i+1],E[i+1]) experiments, assuming the model E=C*h^r
        (for small enough h).
        """
        from math import log
        r = [log(E[i]/E[i-1])/log(h[i]/h[i-1])
            for i in range(1, len(h))]
        return r

    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list

print(fun2(u_1))


