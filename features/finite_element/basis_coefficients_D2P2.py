import numpy as np
import sympy as sym
x, y = sym.symbols('x y')
u = 2*x*x + 3*y*y + 4*x*y + 5*x + 6*y + 7
fun = sym.lambdify([x, y], u, 'numpy')
dudx = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
dudy = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
def grad_fun(x,y):
    return dudx(x,y), dudy(x,y)

grad_fun(1,1)

# 假设有一些数据点 (x, y) 和对应的函数值 f(x, y)
x = np.array([0, 1, 0, 0.5, 0.5, 0])  # x坐标数组
y = np.array([0, 0, 1, 0,   0.5, 0.5])  # y坐标数组

f1 = np.array([1, 0, 0, 0, 0, 0])  # f(x, y) 的值数组
f2 = np.array([0, 1, 0, 0, 0, 0])  # f(x, y) 的值数组
f3 = np.array([0, 0, 1, 0, 0, 0])  # f(x, y) 的值数组
f4 = np.array([0, 0, 0, 1, 0, 0])  # f(x, y) 的值数组
f5 = np.array([0, 0, 0, 0, 1, 0])  # f(x, y) 的值数组
f6 = np.array([0, 0, 0, 0, 0, 1])  # f(x, y) 的值数组

v0 = np.zeros_like(x)
v1 = np.ones_like(x)

# 构建 X 矩阵
X = np.vstack((x*x, y*y, x*y, x, y, v1)).T

# 构建 Y 矩阵
Y = np.vstack((f1, f2, f3, f4, f5, f6)).T

# 使用矩阵求逆计算系数矩阵 [a, b, c]
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

# 提取系数值
for i in range(len(x)):
    print(coefficients[:,i])

def poly_base(x,y):
    return [x*x, y*y, x*y, x, y,1]

def grad_poly_base(x,y):
    return [2*x, 0, y, 1, 0, 0],[0, 2*y, x, 0, 1, 0]

dofs = [fun(point[0], point[1]) for point in zip(x,y)]

def eval(x,y):
    sum = 0
    for i in range(6):
        sum += dofs[i]*np.inner(coefficients[:,i], poly_base(x,y))
    return sum

def eval_grad(x,y):
    sumx = 0
    sumy = 0
    ddddd = grad_poly_base(x,y)
    for i in range(6):
        sumx += dofs[i]*np.inner(coefficients[:,i], ddddd[0])
        sumy += dofs[i]*np.inner(coefficients[:,i], ddddd[1])
    return sumx,sumy


x=0.3
y=0.4
print(eval(x,y), fun(x,y),eval_grad(x,y), grad_fun(x,y))
