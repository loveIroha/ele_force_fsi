# 计算收敛率
DIRICHLET = 1
NEUMANN = 2


def convergence_rates(h, E):
    """
    Given a sequence of discretization parameters in the list h,
    and corresponding errors in the list E,
    compute the convergence rate of two successive (h[i], E[i])
    and (h[i+1],E[i+1]) experiments, assuming the model E=C*h^r
    (for small enough h).
    """
    from math import log
    r = [log(E[i]/E[i-1])/log(h[i]/h[i-1])
        for i in range(1, len(h))]
    return r


import matplotlib.pyplot as plt

def fun_paint(X, Y, Z, title='title', x_axis='x axis', y_axis='y axis', z_axis='z axis', filename='file.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    fig.colorbar(surf)
    plt.title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.view_init(elev=10., azim=-140)
    plt.show()
    plt.savefig(filename)


import sympy as sym
import numpy as np

x, y = sym.symbols('x y')
u_1 = sym.sin(np.pi*x)*sym.sin(np.pi*x)*sym.sin(2*np.pi*y)
u_2 = (x*x-x)*(y*y-y)*sym.exp(x*y)+1
u_3 = (x*x-x)**2*(y*y-y)**2*sym.exp(x*y)+1
u_4 = -x*x*(x-1)*(x-1)*y*(y-1)*(2*y-1)/256+10


def laplace(u):
    return sym.diff(u, x, 2)+sym.diff(u, y, 2)


def grad(u):
    return sym.diff(u, x, 1), sym.diff(u, y, 1)


def make_example(u):
    lambda_u = sym.lambdify([x, y], u, 'numpy')
    lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
    lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
    lambda_laplace_u = sym.lambdify([x, y], laplace(u), 'numpy')
    return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u


def make_examples_helmholtz(u,dt=0.01,mu=0.01,rho=1.0):
    helmholtz = u/dt-mu/rho*laplace(u)
    lambda_u = sym.lambdify([x, y], u, 'numpy')
    lambda_grad_u_x = sym.lambdify([x, y], sym.diff(u, x, 1), 'numpy')
    lambda_grad_u_y = sym.lambdify([x, y], sym.diff(u, y, 1), 'numpy')
    lambda_laplace_u = sym.lambdify([x, y], helmholtz, 'numpy')
    return lambda_u, (lambda_grad_u_x, lambda_grad_u_y), lambda_laplace_u



def make_coordinates(coordinate_x,coordinate_y):
    N = len(coordinate_x)
    M = len(coordinate_y)
    X = np.zeros((N,M))
    Y = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            X[i][j] = coordinate_x[i]
            Y[i][j] = coordinate_y[j]
    return X,Y



def plot_p(Z,Nx,Ny,width=1.0,height=1.0,
           title = "2D Contour Map",
           cmap = 'viridis', x_axis="X-axis", y_axis="Y-axis"):
    dx = width/Nx
    dy = height/Ny
    coordinate_x = np.linspace(-0.5*dx, width+0.5*dx, Nx+2)
    coordinate_y = np.linspace(-0.5*dy, height+0.5*dy, Ny+2)
    X,Y = make_coordinates(coordinate_x,coordinate_y)

    # 绘制等高线图
    plt.contourf(X[1:Nx+1,1:Ny+1], Y[1:Nx+1,1:Ny+1], Z[1:Nx+1,1:Ny+1], cmap=cmap)
    # 马赛克图
    # plt.imshow(Z[1:Nx+1,1:Ny+1], cmap='viridis')

    # 添加颜色条
    plt.colorbar()

    # 设置图形标题和轴标签
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # 显示图形
    plt.savefig(title+".jpg")
    # plt.show()

def plot_p_3D(Z,Nx,Ny,width=1.0,height=1.0,
           title = "2D Contour Map",
           cmap = 'viridis', x_axis="X-axis", y_axis="Y-axis"):
    
    dx = width/Nx
    dy = height/Ny
    coordinate_x = np.linspace(-0.5*dx, width+0.5*dx, Nx+2)
    coordinate_y = np.linspace(-0.5*dy, height+0.5*dy, Ny+2)
    X,Y = make_coordinates(coordinate_x,coordinate_y)

    Zmax = np.max(Z[1:Nx+1,1:Ny+1])
    Zmin = np.min(Z[1:Nx+1,1:Ny+1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    from matplotlib import cm
    surf = ax.plot_surface(X[1:Nx+1,1:Ny+1], Y[1:Nx+1,1:Ny+1], Z[1:Nx+1,1:Ny+1], rstride=1, cstride=1, cmap=cm.jet)
    ax.set_zlim3d(Zmin,Zmax)
    fig.colorbar(surf)
    plt.title('Temperature field in beam cross section')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Temperature')
    ax.view_init(elev=10., azim=-140)
    plt.savefig(title+".jpg")


# plot convergence rates
def plot_convergence_rate(h_list,e_list,r_list,x_axis='$\Delta x$',y_axis='$\|e\|_2$',title='title',legends=[]):
    fig, ax = plt.subplots()
    for i in range(len(h_list)):
        h,e,r = h_list[i], e_list[i], r_list[i]
        print(r)
        plt.plot(h,e,marker="*")

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig(title+'.pdf')



# plot convergence rates
def plot_many_lines(h_list,e_list,x_axis='$\Delta x$',y_axis='$\|e\|_2$',title='title',legends=[]):
    fig, ax = plt.subplots()
    for i in range(len(h_list)):
        h,e = h_list[i], e_list[i]
        plt.plot(h,e,marker="*")

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    ax.set_yscale('linear')
    ax.set_xscale('linear')
    plt.savefig(title+'.pdf')
