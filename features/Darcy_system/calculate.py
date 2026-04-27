from fenics import *
from mshr import *
import ufl
import numpy as np
import sympy as sym

#计算f(J)表达式
J= 2.0
def f(J):
    return 2*(J-1-np.log(J))/(J-1)*(J-1)

A = f(J) 
print('f(J)=', A)

#计算矩阵的行列式，逆矩阵及矩阵的转置
# F = as_matrix([[-1, -2], [3, 4]])    #返回的结果numpy是没法用的
F = np.array([[-1, -2], [3, 4]])
J = np.linalg.det(F)
F_1 = np.linalg.inv(F)
F_T = np.transpose(F_1)
K = 1.0*Identity(2)
print(J)
print(F_1)
print(F_T)
print(J*np.dot(F_1,F_T))

#计算压力p是否正确
M_b = 2.0
b = 1.0
rho = 1.0
kappa = 1.0
phi_0 = 1.0
m = 1.0
J = 2.0
def f(J):
    return 2*(J-1-np.log(J))/(J-1)*(J-1)

def p(m,J):
    return M_b*(b*(1-J) + m/rho)*f(J)-kappa*(rho/(m+rho*phi_0))

B = p(m,J)
print('p(m,J)=', B)


#给定固体的位移函数求解
x, y= sym.symbols('x[0], x[1]')
X = np.array([2*x, y])

def grad(u):
    return sym.Matrix([[sym.diff(u[0], x, 1), sym.diff(u[0], y, 1)], [sym.diff(u[1], x, 1), sym.diff(u[1], y, 1)]])

def grad1(p):
    return sym.Matrix([[sym.diff(p, x, 1)], [sym.diff(p, y, 1)]])

grad_X = grad(X)
print(grad_X)
grad_x = sym.printing.ccode(grad_X)  #将计算机符号转换成C++编码
J = grad_X.det()
print('J=',J)

grad_J = grad1(J)
print('grad_J=', grad_J)

F_1 = grad_X.inv()
print('F_1=', F_1)

F_T = F_1.T
print('F_T=', F_T)

K = 1.0*sym.eye(2)       #sympy中单位矩阵的表示方法
# W = -J*(F_1*K*F_T)
W = -J*F_1*K*F_T
W = sym.simplify(-J*F_1*K*F_T)  #sympy简化表达式利用simplify
print('W=', W)

#前提是给定位移函数求解压力P,其中添加质量M是已知的
def f(J):
    return 2*(J-1-sym.log(J))/(J-1)*(J-1)

M_b = 2.0
b = 1.0
rho = 1.0
kappa = 1.0
phi_0 = 1.0
J = 2.0

x, y, t= sym.symbols('x[0] x[1] t')
M = (1.0-x*x)*(1-t)*t

P = M_b*(b*(1.0-J) + M/rho)*f(J)-kappa*(rho/(M+rho*phi_0))

def grad(u):
    return sym.Matrix([[sym.diff(u, x, 1)], [sym.diff(u, y, 1)]])

def div(u):
    return sym.diff(u[0], x, 1)+sym.diff(u[1], y, 1)

grad_P = grad(P)
print('grad_P =', grad_P)

W = sym.Matrix([[-0.500000000000000, 0], [0, -2.00000000000000]])*grad_P
print('W=', W)
# W = sym.Matrix([[-1.0/(x+1), (x-1)/(x+1)], [(x-1)/(x+1), -(2.0*x*x+2.0)/(x+1)]])*grad_P
print(sym.diff(M,t,1))
S = div(W)+sym.diff(M,t,1)
S_simplify = sym.simplify(S)  
f_code = sym.printing.ccode(S) 
print('S=', f_code)

SS = sym.lambdify((x,y,t),S,'numpy')

for i in range(100):
    print(SS(0,0.5,0.02*i))

