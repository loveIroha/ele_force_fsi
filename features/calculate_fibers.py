import numpy as np


x,y,z = (-9.8,-0.1,-0.1)
r_short_endo = 7
r_short_epi = 10
r_long_endo = 17
r_long_epi = 20
epsilon = 1e-6
alpha_endo, alpha_epi = 90, -90


def transmural_position(x,y,z,xc=0,yc=0,zc=0,endo_short=7,endo_long=17):
    def f(t):
        a = 7+3*t
        b = 7+3*t
        c = 17+3*t
        return b**2*c**2*(x-xc)**2+a**2*c**2*(y-yc)**2+a**2*b**2*(z-zc)**2-a**2*b**2*c**2
    def find_root(f, a, b, eps):
        while b - a > eps:
            c = (a + b) / 2
            if f(c) == 0:
                return c
            elif f(c) * f(a) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2
    
    return find_root(f, 0.0, 1.0, epsilon)


def calculate_fibers(x,y,z):
    t = transmural_position(x,y,z)

    rs = r_short_endo * (1-t) + r_short_epi * t
    rl  = r_long_endo  * (1-t) + r_long_epi  * t
    drs = r_short_epi - r_short_endo
    drl = r_long_epi - r_long_endo

    a = np.sqrt(x*x + y*y) / rs
    b = x / rl

    mu = np.arctan2(a, b)

    theta = 0.0 if mu < epsilon else np.pi - np.arctan2(y, -x)

    sin_m = np.sin(mu)
    cos_m = np.cos(mu)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # 创建一个二维数组
    base = np.array([[drl*cos_m,       -rl*sin_m,        0.0], 
                    [drs*sin_m*cos_t,  rs*cos_m*cos_t, -rs*sin_m*sin_t],
                    [drs*sin_m*sin_t,  rs*cos_m*sin_t,  rs*sin_m*cos_t]])


    if mu < epsilon :
        base = np.array([[1, 0, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])

    # 归一化
    base[:,0] = base[:,0] / np.linalg.norm(base[:,0])
    base[:,1] = base[:,1] / np.linalg.norm(base[:,1])
    base[:,2] = base[:,2] / np.linalg.norm(base[:,2])

    # 强制正交
    base[:,0]  = np.cross(base[:,1] , base[:,2] )

    # 旋转角
    alpha = (alpha_epi - alpha_endo) * t + alpha_endo
    alpha = alpha / 180.0 * np.pi

    import cv2

    # 使用cv2.Rodrigues函数获得旋转矩阵
    rot_mat, _ = cv2.Rodrigues(alpha * base[:,0])

    # 将向量v转换为矩阵形式
    # 计算旋转后的矩阵
    # 将矩阵转换回向量形式
    v_mat = np.reshape(base[:,1], (3, 1))
    v_rot_mat = rot_mat.dot(v_mat)
    base[:,1] = np.reshape(v_rot_mat, (3,))

    v_mat = np.reshape(base[:,2], (3, 1))
    v_rot_mat = rot_mat.dot(v_mat)
    base[:,2] = np.reshape(v_rot_mat, (3,))

    return base


calculate_fibers(x,y,z)
# base = base.colwise().normalized();

# // in general this base is not orthonormal, unless
# //   d/dt ( rs^2(t) - rl^2(t) ) = 0
# bool enforce_orthonormal_base = true;
# if (enforce_orthonormal_base)
# {{
#     base.col(0) = base.col(1).cross(base.col(2));
# }}

# Eigen::Map<mat_type>(values.data()) = base;