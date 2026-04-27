

# sympy 用于符号推导
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
def fun(N=16, max_iters=1000000,u_1=u_1):

    u,f = make_example(u_1)
    width = 1.0
    dx = width/N
    coordiantes = np.linspace(0,1.0,N+1)

    b = np.zeros(N+1)
    phi = np.zeros(N+1)
    u_exact = np.zeros(N+1)
    error = np.zeros(N+1)

    for i in range(N+1):
        u_exact[i] = u(coordiantes[i])
        b[i] = f(coordiantes[i])


    for iter in range(max_iters):
        for i in range(1,N):
            phi[i] = (phi[i+1]+phi[i-1]-dx*dx*b[i-1])/2


    for i in range(1,N+1):
        error[i] = phi[i] - u_exact[i]

    error_norm_l2 = np.sum(error**2)*dx
    return error_norm_l2




E_list = []
h_list = []
N_list = []
for i in range(3,10):
    N = 1<<i
    h = 1.0/N
    E = fun(N,1000000,u_2)
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

print(convergence_rates(h_list, E_list))



legends = []
fig, ax = plt.subplots()
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

plt.plot(h_list,E_list,marker='*', linewidth=0.5)
legends.append(str('%s $N$ = %3.1f' % ('$u_1$', N)))
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend(legends)
plt.xlabel('h')
plt.ylabel('$\|e\|_2$')
plt.show()
