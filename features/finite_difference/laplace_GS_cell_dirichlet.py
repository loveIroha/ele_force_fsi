


import sympy as sym 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

x = sym.symbols('x') 
u_1 = sym.sin(np.pi*x)
u_2 = sym.exp(x)*(x*x-x)
u_3 = (x*x-x)**3+1

def make_example(u):
    def laplace(u): 
        return sym.diff(u, x, 2)
    lambda_u =  sym.lambdify(x, u, 'numpy')
    lambda_laplace_u = sym.lambdify(x, laplace(u), 'numpy')
    return lambda_u, lambda_laplace_u

DIRICHLET = 1
NEUMANN= 2

# discretizing geometry
def fun(N=16, max_iters=100,u_1=u_1):

    u,f = make_example(u_1)
    width = 1.0
    dx = width/N
    coordiantes = np.linspace(0-0.5*dx,1.0+0.5*dx,N+2)

    b = np.zeros(N+2)
    phi = np.zeros(N+2)
    u_exact = np.zeros(N+2)
    error = np.zeros(N+2)

    boundary_type = np.zeros(N+2)
    boundary_type[0] = DIRICHLET
    boundary_type[N+1] = DIRICHLET

    for i in range(N+2):
        u_exact[i] = u(coordiantes[i])
        b[i] = f(coordiantes[i])


    for iter in range(max_iters):

        # 处理边界条件
        if boundary_type[0] == DIRICHLET:
            phi[0]=2*u(0.0)-phi[1]
        if boundary_type[N+1] == DIRICHLET:
            phi[N+1]=2*u(1.0)-phi[N]
        # print(phi[0],phi[1],u(0.0),u_exact[0],u_exact[1])
        # print(phi[N],phi[N+1],u(1.0),u_exact[N],u_exact[N+1])
        
        # 迭代求解
        for i in range(1,N+1):
            phi[i] = (phi[i+1]+phi[i-1]-dx*dx*b[i-1])/2


    for i in range(1,N+1):
        error[i] = phi[i] - u_exact[i]

    error_norm_l2 = np.sum(error**2)*dx
    return error_norm_l2




def fun2(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(3,8):
        N = 1<<i
        h = 1.0/N
        E = fun(N,100000,u)
        E_list.append(E)
        h_list.append(h)
        N_list.append(N)


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


def fun3():
    legends=[]
    fig, ax = plt.subplots()
    h,e,r = fun2(u_1)
    print(r)
    plt.plot(h,e,marker='*')
    legends.append(str('$u_1$'))
    h,e,r = fun2(u_2)
    print(r)
    plt.plot(h,e,marker='+')
    legends.append(str('$u_2$'))  
    h,e,r = fun2(u_3)
    print(r)
    plt.plot(h,e,marker='o')
    legends.append(str('$u_3$'))   
    plt.legend(legends)
    plt.xlabel('$\Delta x$')
    plt.ylabel('$\|e\|_2$')
    plt.title('convergence rate')

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

fun3()
