import numpy as np
import sympy as sym
x, y, z = sym.symbols('x y z')
u = 2*x*x + 3*y*y + z*z + y*z + x*z +4*x*y + 5*x + 6*y + z + 7
fun = sym.lambdify([x, y, z], u, 'numpy')
dudx = sym.lambdify([x, y, z], sym.diff(u, x, 1), 'numpy')
dudy = sym.lambdify([x, y, z], sym.diff(u, y, 1), 'numpy')
dudz = sym.lambdify([x, y, z], sym.diff(u, z, 1), 'numpy')
def grad_fun(x,y,z):
    return dudx(x,y,z), dudy(x,y,z), dudz(x,y,z)

grad_fun(1,1,1)

# 假设有一些数据点 (x, y) 和对应的函数值 f(x, y)
x = np.array([0, 1, 0, 0, 0.5, 0,   0,   0.5, 0.5, 0])  # x坐标数组
y = np.array([0, 0, 1, 0, 0,   0.5, 0,   0.5, 0,   0.5])  # y坐标数组
z = np.array([0, 0, 0, 1, 0,   0,   0.5, 0,   0.5, 0.5])  # y坐标数组

f0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f4 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f5 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # f(x, y) 的值数组
f6 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # f(x, y) 的值数组
f7 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # f(x, y) 的值数组
f8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])  # f(x, y) 的值数组
f9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # f(x, y) 的值数组

v0 = np.zeros_like(x)
v1 = np.ones_like(x)

# 构建 X 矩阵
X = np.vstack((x*x, y*y, z*z, y*z, x*z, x*y, x, y, z, v1)).T

# 构建 Y 矩阵
Y = np.vstack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9)).T

# 使用矩阵求逆计算系数矩阵 [a, b, c]
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

# 提取系数值
for i in range(len(x)):
    print(coefficients[:,i])

def poly_base(x,y,z):
    return [x*x, y*y, z*z, y*z, x*z, x*y, x, y, z, 1]

def grad_poly_base(x,y,z):
    return [2*x, 0,   0,   0, z, y, 1, 0, 0, 0], [0,   2*y, 0,   z, 0, x, 0, 1, 0, 0], [0,   0,   2*z, y, x, 0, 0, 0, 1, 0]

dofs = [fun(point[0], point[1], point[2]) for point in zip(x,y,z)]

def eval(x,y,z):
    sum = 0
    for i in range(10):
        sum += dofs[i]*np.inner(coefficients[:,i], poly_base(x,y,z))
    return sum

def eval_grad(x,y,z):
    sumx = 0
    sumy = 0
    sumz = 0
    ddddd = grad_poly_base(x,y,z)
    for i in range(10):
        sumx += dofs[i]*np.inner(coefficients[:,i], ddddd[0])
        sumy += dofs[i]*np.inner(coefficients[:,i], ddddd[1])
        sumz += dofs[i]*np.inner(coefficients[:,i], ddddd[2])
    return sumx,sumy,sumz


x=0.3
y=0.4
z= 0.5
print(eval(x,y,z), fun(x,y,z),eval_grad(x,y,z), grad_fun(x,y,z))

