


# 读入网格
from fenics import *
mesh_in = Mesh()
f_mesh_in = XDMFFile("/home/kokkos/geometry/ventricle/LV/LV_ideal/LV_ideal_30.xdmf")
f_mesh_in.read(mesh_in)
f_mesh_in.close()
  
# 读入函数
W = VectorFunctionSpace(mesh_in, 'P', 1)
w = Function(W)
f_in = XDMFFile(mesh_in.mpi_comm(), "/home/kokkos/ssh/npuheart/build_ilv_systole/ideal_LV_diastole_3D_implicit_30_64_16000_2.0_with_4.00e-02_1.00e+06_5.00e+07/solid/process.xdmf")
# f_in = XDMFFile(mesh_in.mpi_comm(), "/home/kokkos/ssh/npuheart/build_ilv_systole/ideal_LV_diastole_3D_implicit_30_64_16000_2.0_with_4.00e-02_1.00e+06_5.00e+07/solid/process.xdmf")
f_in.read_checkpoint(w,"displacement",0)

# 定义参考位置和当前位置
W2 = VectorFunctionSpace(mesh_in, 'P', 1)
position_ref = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=1), W2)
position_current = Function(W2)
position_current.vector()[:] = position_ref.vector()[:] + w.vector()[:]
position_current.set_allow_extrapolation(True)
position_ref.set_allow_extrapolation(True)

# 计算三条曲线上的点
# 计算三条曲线上的点
import numpy as np
from numpy import cos, sin, arccos
positions_t = []
n_step = 1000
v = 0
u = lambda i, t: -np.pi + i * (-np.arccos(5/(17+t*3)) + np.pi)
r_s = lambda t: 7 + t*3
r_l = lambda t: 17 + t*3
for t in [0.5]:
    uvts = [(  u(i / n_step, t),v,t) for i in range(n_step)]
    positions = [(r_s(t)*sin(u)*cos(v), r_s(t)*sin(u)*sin(v), r_l(t)*cos(u)) for (u,v,t) in uvts]
    positions = np.array(positions)
    positions_t.append(positions)

X = [positions_t[0][:,0]/10+2.5]
Y = [positions_t[0][:,1]/10+2.5]
Z = [positions_t[0][:,2]/10+3.5]
x1 = [position_current(Z[0][i], Y[0][i], X[0][i])[0] for i in range(0, n_step)]
y1 = [position_current(Z[0][i], Y[0][i], X[0][i])[2] for i in range(0, n_step)]

# import numpy as np
# from numpy import cos, sin, arccos
# n_step = 1000
# v = 0
# t = 0.5
# u = lambda i, t: -np.pi + i * (-np.arccos(5/(17+t*3)) + np.pi)
# r_s = lambda t: 7 + t*3
# r_l = lambda t: 17 + t*3
# uvts = [(  u((i) / n_step, t),v,t) for i in range(n_step)]
# positions = [(r_s(t)*sin(u)*cos(v), r_s(t)*sin(u)*sin(v), r_l(t)*cos(u)) for (u,v,t) in uvts]
# positions = np.array(positions)

# X1 = [position_current(positions[i][2]/10+2.5, 2.5, positions[i][0]/10+3.5)[0] for i in range(0, n_step)]
# Y1 = [position_current(positions[i][2]/10+2.5, 2.5, positions[i][0]/10+3.5)[2] for i in range(0, n_step)]
# X1 = [positions[i][0]/10+3.5 for i in range(0, n_step)]
# Y1 = [positions[i][2]/10+2.5 for i in range(0, n_step)]
# X1 = [position_ref(positions[i][2]/10+2.5, 2.5, positions[i][0]/10+3.5)[0] for i in range(0, n_step)]
# Y1 = [position_ref(positions[i][2]/10+2.5, 2.5, positions[i][0]/10+3.5)[2] for i in range(0, n_step)]

from matplotlib import pyplot as plt
def plot(h_list,e_list,x_axis='$\Delta x$',y_axis='$\|e\|_2$',title='title',markers=['.','.'],legends=[]):
    # fig, ax = plt.subplots()
    for i in range(len(h_list)):
        h,e = h_list[i], e_list[i]
        plt.plot(h,e,marker=markers[i%len(markers)], linestyle='dashed', linewidth=1.5)
    plt.legend(legends)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    # plt.axis('equal')
    plt.xlim(1.0,1.25)
    plt.ylim(2.6,3.3)

plt.figure()
plot(X,Z,x_axis='x',y_axis='y',title='t=',legends=['t=0.1','t=0.5','t=0.9'])
plot([y1],[x1],x_axis='x',y_axis='y',title='t=',legends=['t=0.1','t=0.5','t=0.9'])
plt.savefig('t='+str(t)+'area.png',dpi=300)
plt.close()


