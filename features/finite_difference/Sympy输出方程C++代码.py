
import sympy as sym
from sympy import sin, cos, pi, exp


x, y, t = sym.symbols('x[0], x[1], t') 
rho, mu = sym.symbols('rho, mu')

def q(u):
    return sym.sin(u)


def cpp_code(f, name='f'):
    f = sym.simplify(f) 
    f_code = sym.printing.ccode(f) 
    return 'double ' + name + ' = ' + f_code + ";"


# 压强可不可以替换？
# p = sym.exp(t)*((x*x-x)*(y*y-y))


u = - sym.exp(t)*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256
v = sym.exp(t)*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)/256
p = exp(t)*(x*x*x-0.25)

f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))

print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))

u = sym.exp(t)*sym.sin(pi*x)**2*sym.sin(2*pi*y)
v = -sym.exp(t)*sym.sin(2*pi*x)*sym.sin(pi*y)**2
p = exp(t)*(sin(pi*y)-2/pi)

f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))

print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))

u = 20*x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)*t
v = -20*x*(x-1)*(2*x-1)*y*y*(y-1)*(y-1)*t
p = 10*(2*x-1)*(2*y-1)
f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))
print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))

u =  2*pi*sin(pi*x)**2*sin(pi*y)*cos(pi*y)*cos(t)
v = -2*pi*sin(pi*x)*cos(pi*x)*sin(pi*y)**2*cos(t)
p = cos(pi*x)*cos(pi*y)
f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))
print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))

u = 2*cos(pi*y)*sin(pi*x)*sin(t)
v = -2*sin(pi*y)*cos(pi*x)*sin(t)
p = 2*sin(pi*y)*cos(pi*x)*cos(t)
f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))
print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))


u = (x*x*y*y+exp(-y))*cos(2*pi*t)
v = (-2/3*x*y*y*y+2-pi*sin(pi*x))*cos(2*pi*t)
p = -(2-pi*sin(pi*x))*cos(2*pi*y)*cos(2*pi*t)
f1 = rho*sym.diff(u, t, 1) - mu*(sym.diff(u, x, 2) + sym.diff(u, y, 2)) + sym.diff(p, x, 1)
f2 = rho*sym.diff(v, t, 1) - mu*(sym.diff(v, x, 2) + sym.diff(v, y, 2)) + sym.diff(p, y, 1)
print(cpp_code(f1,'f1'))
print(cpp_code(f2,'f2'))
print(sym.simplify(sym.diff(u,x,1)+sym.diff(v,y,1)))

print(cpp_code(u))
