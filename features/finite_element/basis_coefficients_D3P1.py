import numpy as np

# 假设有一些数据点 (x, y) 和对应的函数值 f(x, y)
x = np.array([0, 1, 0, 0])  # x坐标数组
y = np.array([0, 0, 1, 0])  # y坐标数组
z = np.array([0, 0, 0, 1])  # y坐标数组

f0 = np.array([1, 0, 0, 0])  # f(x, y) 的值数组
f1 = np.array([0, 1, 0, 0])  # f(x, y) 的值数组
f2 = np.array([0, 0, 1, 0])  # f(x, y) 的值数组
f3 = np.array([0, 0, 0, 1])  # f(x, y) 的值数组


v0 = np.zeros_like(x)
v1 = np.ones_like(x)

# 构建 X 矩阵
X = np.vstack((x, y, z, v1)).T
# grad_x = np.vstack((v1, v0, v0, v0)).T
# grad_y = np.vstack((v0, v1, v0, v0)).T
# grad_z = np.vstack((v0, v0, v1, v0)).T

# 构建 Y 矩阵
Y = np.vstack((f0, f1, f2, f3)).T

# 使用矩阵求逆计算系数矩阵 [a, b, c]
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

# 提取系数值
for i in range(len(x)):
    print(coefficients[:,i])
    # print(grad_x*coefficients[:,i])
    # print(grad_y*coefficients[:,i])
    # print(grad_z*coefficients[:,i])
