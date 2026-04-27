


import sympy as sym 

import numpy as np
import matplotlib.pyplot as plt

x = sym.symbols('x') 
u_1 = sym.sin(np.pi*x)
u_2 = sym.exp(x)*(x*x-x)
u_3 = (x*x-x)**3

def make_example(u):
    def laplace(u): 
        return sym.diff(u, x, 2)
    lambda_u =  sym.lambdify(x, u, 'numpy')
    lambda_laplace_u = sym.lambdify(x, laplace(u), 'numpy')
    return lambda_u, lambda_laplace_u



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

    for i in range(N+1):
        u_exact[i] = u(coordiantes[i])
        b[i] = f(coordiantes[i])


    for iter in range(max_iters):
        phi[0]=-phi[1]
        phi[N]=-phi[N+1]
        for i in range(1,N):
            phi[i] = (phi[i+1]+phi[i-1]-dx*dx*b[i-1])/2


    for i in range(1,N+1):
        error[i] = phi[i] - u_exact[i]

    error_norm_l2 = np.sum(error**2)*dx
    return error_norm_l2




def fun2(u):
    E_list = []
    h_list = []
    N_list = []
    for i in range(3,10):
        N = 1<<i
        h = 1.0/N
        E = fun(N,100000,u)
        E_list.append(E)
        h_list.append(h)
        N_list.append(N)




    R_list = convergence_rates(h_list, E_list)

    return h_list, E_list, R_list


def fun3():
    legends=[]
    fig, ax = plt.subplots()
    h,e,r = fun2(u_1)
    plt.plot(h,e,marker='*')
    legends.append(str('$u_1$'))
    h,e,r = fun2(u_2)
    plt.plot(h,e,marker='+')
    legends.append(str('$u_2$'))  
    h,e,r = fun2(u_3)
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
