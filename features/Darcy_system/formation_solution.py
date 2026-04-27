import sympy as sym
import numpy as np


def f(J):
    return 2*(J-1-sym.log(J))/(J-1)*(J-1)

M_b = 2.0
b = 1.0
rho = 1.0
kappa = 1.0
phi_0 = 1.0
J = 2.0

x, y, t= sym.symbols('x[0] x[1] t')
M = 1.0+x*x + 2.0*y*y + 3.0*t

P = M_b*(b*(1.0-J) + M/rho)*f(J)-kappa*(rho/(M+rho*phi_0))

def grad(u):
    return sym.Matrix([[sym.diff(u, x, 1)], [sym.diff(u, y, 1)]])

def div(u):
    return sym.diff(u[0], x, 1)+sym.diff(u[1], y, 1)

grad_P = grad(P)

W = sym.Matrix([[-10.0, 7.0], [7.0, -5.0]])*grad_P
S = div(W)+3.0

f_code = sym.printing.ccode(S) 

# u_1 = sym.sin(np.pi*x)*sym.sin(np.pi*x)*sym.sin(2*np.pi*y) + 1 
# u_2 = (x*x-x)*(y*y-y)*sym.exp(x*y)
# u_3 = (x*x-x)**2*(y*y-y)**2*sym.exp(x*y)+1
# u_4 = -x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256+10



# def laplace(u):
#     return sym.diff(u, x, 2)+sym.diff(u, y, 2)


# def grad(u):
#     return sym.diff(u, x, 1), sym.diff(u, y, 1)

# # 打印C++代码
# def cpp_code(f, name='f'):
#     f = sym.simplify(f) 
#     f_code = sym.printing.ccode(f) 
#     return 'double ' + name + ' = ' + f_code + ";"

# def make_example(u):
#     lambda_u = sym.lambdify([x, y], u, 'numpy')
#     lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
#     lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
#     lambda_laplace_u = sym.lambdify([x, y], laplace(u), 'numpy')
#     return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u


# def make_examples_helmholtz(u,dt=0.01,mu=0.01,rho=1.0):
#     helmholtz = u/dt-mu/rho*laplace(u)
#     # 打印出右端项
#     print(cpp_code(u,'u'))
#     print(cpp_code(helmholtz,'f'))
#     print(cpp_code(sym.diff(u, x, 1),'dudx'))
#     print(cpp_code(sym.diff(u, y, 1),'dudy'))
#     lambda_u = sym.lambdify([x, y], u, 'numpy')
#     lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
#     lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
#     lambda_laplace_u = sym.lambdify([x, y], helmholtz, 'numpy')
#     return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u