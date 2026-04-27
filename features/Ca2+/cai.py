import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件
file_path = 'a.txt'  # 请替换为你的文件路径
data = np.loadtxt(file_path)

# 提取列数据
x_data = data[:, 0]
y_data = data[:, 1]

for i in range(1, len(x_data)):
    print(x_data[i] - x_data[i-1])

# 绘制折线图
plt.plot(x_data, y_data, label='Data')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot of Data')
plt.legend()
plt.savefig("b.png")
