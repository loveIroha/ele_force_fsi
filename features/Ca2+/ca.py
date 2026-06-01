import numpy as np
import matplotlib.pyplot as plt



# 初始条件
t = 0


# 时间离散
Nt = 10000
dt = 1.0 / Nt
t_array = np.linspace(0, 1, num=Nt+1)



# 第一步：计算钙离子浓度$ [Ca]_{i} $:


# 常数
Ca_i_max = 1.0
Ca_i_0 = 0.01
tau_Ca = 0.06

# 未知量
Ca_i = np.zeros(Nt+1)
Ca_i[0] = 0.0   # TODO : 查一下初始值

def Ca_i_expression(i):
    t = t_array[i]
    return (Ca_i_max - Ca_i_0) / tau_Ca * np.exp(1 - t / tau_Ca) * (1 - t / tau_Ca)


for i in range(1, Nt+1):
    t = t_array[i]
    Ca_i[i] = Ca_i_expression(i)*dt + Ca_i[i-1]


# 绘制曲线
# plt.plot(t_array, Ca_i, label='[Ca] vs Time')
# plt.xlabel('Time (t)')
# plt.ylabel('[Ca]')
# plt.title('Calcium Concentration Over Time')
# plt.ylim(0, 1)
# plt.xlim(0, 1)
# plt.legend()

# plt.savefig('ca.png')



# 第二步：计算肌钙蛋白的钙离子浓度


# 常数

T_ref = 56.2 # kPa
gamma = 2.0
k_on = 100             # u/M/s
k_off_ref = 200        # /s
Ca_trpn_max = 70

# 中间变量
def k_off(T):
    return k_off_ref * np.exp(1 - T/(gamma*T_ref))

# 未知量
Ca_trpn = np.zeros(Nt+1)
T = np.zeros(Nt+1)
T[0] = 0.0



# 第三步：计算半激活状态下肌钙蛋白结合的钙离子浓度$[Ca]_{trpn,50}$

Ca_50_ref = 1.05 # uM
beta_1 = -4.0    
beta_0 = 4.9

lambda_f = np.zeros(Nt+1)
for i in range(0, Nt+1):
    lambda_f[i] = 1.15

Ca_trpn_50 = np.zeros(Nt+1)



# 第四步：原肌球蛋白钙离子浓度 $z$ 的计算
n = 3
n_r = 3
z_p = 0.85 
K_z = 0.15
alpha_0 = 8   # /s
alpha_1 = 30 # /s
alpha_2 = 130 # /s
alpha_3 = 625 # /s
alpha_r1 = 2.0 # /s
alpha_r2 = 1.75 # /s


z = np.zeros(Nt+1)


def z_expression(i):
    temp_1 = alpha_0* (Ca_trpn[i]/Ca_trpn_50[i])**n*(1-z[i-1])
    temp_2 = - alpha_r1*z[i-1] - alpha_r2*(z[i-1]**n_r)/(z[i-1]**n_r+K_z**n_r)
    return  temp_1 + temp_2



# 第五步：等容积收缩张力

K_1 = (alpha_r2*z_p**(n_r-1)*n_r*K_z**n_r)/(z_p**n_r + K_z**n_r)**2
K_2 = alpha_r2*z_p**n_r/(z_p**n_r+K_z**n_r)*(1-n_r*K_z**n_r/(z_p**n_r+K_z**n_r))

z_max = np.zeros(Nt+1)
T0 = np.zeros(Nt+1)



# 第六步： 主动收缩张力$T$ 是通过一个衰减记忆模型计算得出

A_1 = -29
A_2 = 138
A_3 = 129
a = 0.35


Q_1 = np.zeros(Nt+1)
Q_2 = np.zeros(Nt+1)
Q_3 = np.zeros(Nt+1)
Q_sum = np.zeros(Nt+1)


d_lamdba_f = np.zeros(Nt+1)

for i in range(2, Nt+1):
    # d_lamdba_f[i] = (lambda_f[i-1] - lambda_f[i-2])/dt
    d_lamdba_f[i] = -0.15
    
for i in range(2, Nt+1):
    Q_1[i] = (A_1 * d_lamdba_f[i] - alpha_1*Q_1[i-1])*dt + Q_1[i-1]
    Q_2[i] = (A_2 * d_lamdba_f[i] - alpha_2*Q_2[i-1])*dt + Q_2[i-1]
    Q_3[i] = (A_3 * d_lamdba_f[i] - alpha_3*Q_3[i-1])*dt + Q_3[i-1]
    Q_sum[i] = Q_1[i] + Q_2[i] + Q_3[i]
    print("Q_sum : ", Q_sum[i])


        
# # 绘制曲线
# plt.plot(t_array, Q_sum, label='[Q_sum] vs Time')
# plt.xlabel('Time (t)')
# plt.ylabel('[Ca]')
# plt.title('Calcium Concentration Over Time')
# plt.ylim(0, 1)
# plt.xlim(0, 1)
# plt.legend()

# plt.savefig('ca.png')



for i in range(1, Nt+1):
    t = t_array[i]
    
    def Ca_trpn_expression(i):
        return k_on * Ca_i[i] * (Ca_trpn_max - Ca_trpn[i-1]) - k_off(T[i-1]) * Ca_trpn[i-1]

    Ca_trpn[i] = Ca_trpn_expression(i)*dt + Ca_trpn[i-1]

    
    def Ca_50(lambda_f):
        return Ca_50_ref*(1+beta_1*(lambda_f-1))

    temp_1 =k_off_ref / k_on * (1-(1+beta_0*(lambda_f[i-1]-1))/2.0/gamma)

    Ca_trpn_50[i] = Ca_trpn_max*Ca_50(lambda_f[i-1])/(Ca_50(lambda_f[i-1]) + temp_1)
    
    z[i] = z_expression(i)*dt + z[i-1]
    frac_1 = (alpha_0/((Ca_trpn_50[i]/Ca_trpn_max)**n)-K_2)
    frac_2 = (alpha_r1+K_1 + alpha_0/((Ca_trpn_50[i]/Ca_trpn_max)**n))
    z_max[i] = frac_1 / frac_2
    T0[i] = T_ref * (1 + beta_0*(lambda_f[i-1]-1))*z[i]/z_max[i]
    if Q_sum[i] < 0:
        T[i] = T0[i]*(1+a*Q_sum[i])/(1 - Q_sum[i])
    else:
        T[i] = T0[i]*(1+(2+a)*Q_sum[i])/(1 + Q_sum[i])
        
# 绘制曲线
plt.plot(t_array, T, label='[Q_sum] vs Time')
plt.xlabel('Time (t)')
plt.ylabel('[Ca]')
plt.title('Calcium Concentration Over Time')
plt.ylim(0, 15)
plt.xlim(0, 1)
plt.legend()

plt.savefig('ca.png')