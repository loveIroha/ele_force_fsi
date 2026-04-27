import numpy as np


def f(x,y,z):
    return x**2 + y**2 + z**2


# 假设有一些数据点 (x, y) 和对应的函数值 f(x, y)
x = np.array([0, 0, 1])  # x坐标数组
y = np.array([0, 1, 0])  # y坐标数组
f1_values = np.array([1, 0, 0])  # f(x, y) 的值数组
f2_values = np.array([0, 1, 0])  # f(x, y) 的值数组
f3_values = np.array([0, 0, 1])  # f(x, y) 的值数组

# 构建 X 矩阵
X = np.vstack((x, y, np.ones_like(x))).T

# 构建 Y 矩阵
Y = np.vstack((f1_values, f2_values, f3_values)).T

# 使用矩阵求逆计算系数矩阵 [a, b, c]
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

# 提取系数值
a, b, c = coefficients.ravel()

# 打印结果
print(f"a = {a}, b = {b}, c = {c}")
