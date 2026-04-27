import numpy as np
import matplotlib.pyplot as plt

def loading_pressure(time):
    P_load = 1.0  # 你可能需要根据实际情况调整这个值
    t_load = 0.5  # 你可能需要根据实际情况调整这个值
    t_end_diastole = 1.0  # 你可能需要根据实际情况调整这个值

    if time <= t_load:
        P = P_load * time / t_load
    elif t_load < time <= t_end_diastole:
        P = P_load
    elif t_end_diastole < time <= t_end_diastole + 0.2:
        P = P_load * 17.75 * ((time - t_end_diastole) / 0.2 if time <= t_end_diastole + 0.2 else 1.0) + P_load
    else:
        P = 18.75 * P_load

    return P*1333.223684

# 生成时间数组
time_array = np.linspace(0, 2, num=1000)  # 你可能需要根据实际情况调整这个时间范围和数量

# 计算对应时间的压力值
pressure_array = np.vectorize(loading_pressure)(time_array)

# 绘制压力与时间的图形
plt.plot(time_array, pressure_array, label='Pressure vs Time')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Loading Pressure Function')
plt.legend()
plt.savefig("a.png")
