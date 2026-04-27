$(N_{i,}R_{i}),\;i=0,\;\dots,\;m-1$为一系列实验数据，其中$N_i$为网格单元数量，$R_i$为误差，收敛率(convergence rate)$r_{i-1}$，可从相邻两个实验数据中计算出来：
$$
r_{i-1}=\frac{\ln(R_{i-1}/R_{i})}{\ln(\Delta x_{i-1}/\Delta x_{i})},
$$
$L_2$误差的计算公式为
$$
R_i=\Delta x_i\sum_{k=1}^{N_i+1}(e_k)^2.
$$

<img src="https://githubimages.pengfeima.cn/images/202305272126725.jpg" alt="交错网格" style="zoom:25%;" />





<img src="https://githubimages.pengfeima.cn/images/202305272241795.jpg" alt="1951685198434_.pic" style="zoom:25%;" />





# 泊松方程


$$
-\frac{u_{i-1, j}-2 u_{i, j}+u_{i+1, j}}{h_x^2}-\frac{u_{i, j-1}-2 u_{i, j}+u_{i, j+1}}{h_y^2}=f_{i, j},\tag{1}
$$

$$
-\frac{u_{i-\frac{3}{2}, j}-2 u_{i-\frac{1}{2}, j}+u_{i+\frac{1}{2}, j}}{h_x^2}-\frac{u_{i-\frac{1}{2}, j-1}-2 u_{i-\frac{1}{2}, j}+u_{i-\frac{1}{2}, j+1}}{h_y^2}=f_{i-\frac{1}{2}, j},\tag{2}
$$

$$
-\frac{u_{i-1, j-\frac{1}{2}}-2 u_{i, j-\frac{1}{2}}+u_{i+1, j-\frac{1}{2}}}{h_x^2}-\frac{u_{i, j-\frac{3}{2}}-2 u_{i, j-\frac{1}{2}}+u_{i, j+\frac{1}{2}}}{h_y^2}=f_{i, j-\frac{1}{2}},\tag{3}
$$

$$
-\frac{u_{i-\frac{3}{2}, j-\frac{1}{2}}-2 u_{i-\frac{1}{2}, j-\frac{1}{2}}+u_{i+\frac{1}{2}, j-\frac{1}{2}}}{h_x^2}-\frac{u_{i-\frac{1}{2}, j-\frac{3}{2}}-2 u_{i-\frac{1}{2}, j-\frac{1}{2}}+u_{i-\frac{1}{2}, j+\frac{1}{2}}}{h_y^2}=f_{i-\frac{1}{2}, j-\frac{1}{2}},\tag{4}
$$





<img src="https://githubimages.pengfeima.cn/images/202305251034404.png" alt="image-20230525103403246" style="zoom:25%;" />

## 2. 速度$u$的求解

 $u$ 分量沿 $y$ 方向向后偏移了$\frac{1}{2}$个网格步长，可得到方程
$$
u_{i, j-\frac{1}{2}}=\frac{(u_{i-1, j-\frac{1}{2}}+u_{i+1, j-\frac{1}{2}})h_y^2+(u_{i, j-\frac{3}{2}}+u_{i, j-\frac{1}{2}})h_x^2-h_x^2h_y^2f_{i, j-\frac{1}{2}}}{2h_y^2+2h_x^2},
$$
$v$的未知量个数为$(N_x+1)\times(N_y+2)$，包含$y$方向上下两边的虚拟点。在**正则内点**$i\in[1,N_x-1],j\in[1,N_y]$对微分方程使用**标准的五点差分格式(standard 5-point finite difference stencil)**，区域的四条边需要单独处理，四个角点归入上下两条边界处理。

在处理边界条件时，左右边界的角点需要知道上下边界的未知量，而上下边界不需要知道左右边界的未知量，因此先要遍历上下边界，再便利左右边界。

1. 上边界：$v_{i,-\frac{1}{2}}$，$i\in[0,N_x]$

   **Nevmann边界条件**：将$\frac{\partial v}{\partial y}=g$进行中心差分离散，可得$\frac{v_{i,\frac{1}{2}}-v_{i,-\frac{1}{2}}}{h_y}=g_{i,0}$ , 即

$$
v_{i,-\frac{1}{2}}=v_{i,\frac{1}{2}}-h_yg_{i,0}
$$

​       **Dirichlet边界条件**：$v_{i,\frac{1}{2}}+v_{i,-\frac{1}{2}}=2g_{i,0}$，即
$$
v_{i,-\frac{1}{2}}=2g_{i,0}-v_{i,\frac{1}{2}}
$$

2. 下边界：$v_{i,N_y+\frac{1}{2}}$，$i\in[0,N_x]$

​       **Nevmann边界条件**：类似可得$\frac{v_{i,N_{y}+\frac{1}{2}}-v_{i,N_y-\frac{1}{2}}}{h_y}=g_{i,N_y}$, 即
$$
v_{i,N_y+\frac{1}{2}}=v_{i,N_y-\frac{1}{2}}+h_yg_{i,N_y}
$$

​       **Dirichlet边界条件**：$v_{i,N_{y}+\frac{1}{2}}+v_{i,N_y-\frac{1}{2}}=2g_{i,N_y}$
$$
v_{i,N_{y}+\frac{1}{2}}=2g_{i,N_y}-v_{i,N_y-\frac{1}{2}}
$$

3. 左边界：$v_{0,j-\frac{1}{2}}$，$j\in[1,N_y]$

   **Nevmann边界条件**：$\frac{v_{1,j-\frac{1}{2}}-v_{-1,j-\frac{1}{2}}}{2h_x}=g_{0,j-\frac{1}{2}}$
   $$
   v_\text{ghost}={v_{-1,j-\frac{1}{2}}}=v_{1,j-\frac{1}{2}}-2h_xg_{0,j}
   $$

   $$
   u_{0, j-\frac{1}{2}}=\frac{(u_\text{ghost}+u_{1, j-\frac{1}{2}})h_y^2+(u_{0, j-\frac{3}{2}}+u_{0, j-\frac{1}{2}})h_x^2-h_x^2h_y^2f_{0, j-\frac{1}{2}}}{2h_y^2+2h_x^2},
   $$

   

​		**Dirichlet边界条件**：$u_{0, j-\frac{1}{2}}=g_{0,j-\frac{1}{2}}$



4. 右边界：$v_{N_x,j-\frac{1}{2}}$，$j\in[1,N_y]$

​		**Nevmann边界条件**：$\frac{v_{N_x+1,j-\frac{1}{2}}-v_{N_x-1,j-\frac{1}{2}}}{2h_x}=g_{N_x,j-\frac{1}{2}}$
$$
v_\text{ghost}={v_{N_x+1,j-\frac{1}{2}}}=v_{N_x-1,j-\frac{1}{2}}+2h_xg_{N_x,j}
$$

$$
u_{N_x, j-\frac{1}{2}}=\frac{(u_{N_x+1, j-\frac{1}{2}}+u_\text{ghost})h_y^2+(u_{N_x, j-\frac{3}{2}}+u_{N_x, j-\frac{1}{2}})h_x^2-h_x^2h_y^2f_{N_x, j-\frac{1}{2}}}{2h_y^2+2h_x^2},
$$

​		**Dirichlet边界条件**：$u_{N_x, j-\frac{1}{2}}=g_{N_x,j-\frac{1}{2}}$















### 压强 $p$ 的求解



<img src="https://githubimages.pengfeima.cn/images/202305251034404.png" alt="image-20230525103403246" style="zoom:25%;" />

Poisson方程为
$$
\Delta u=f,
$$

逼近Poisson方程的五点差分格式为
$$
\frac{u_{i-1, j}-2 u_{i, j}+u_{i+1, j}}{h_x^2}+\frac{u_{i, j-1}-2 u_{i, j}+u_{i, j+1}}{h_y^2}=f_{i, j},
$$

可以根据下面的等式
$$
u_{i, j}=\frac{(u_{i-1, j}+u_{i+1, j})h_y^2+(u_{i, j-1}+u_{i, j+1})h_x^2-h_x^2h_y^2f_{i, j}}{2h_y^2+2h_x^2},
$$
构造迭代格式进行求解。



## 2. 速度$v$的求解

 $v$ 分量沿 $x$ 方向向后偏移了$\frac{1}{2}$个网格步长，可得到方程
$$
v_{i-\frac{1}{2}, j}=\frac{(v_{i-\frac{3}{2}, j}+v_{i+\frac{1}{2}, j})h_y^2+(v_{i-\frac{1}{2}, j-1}+v_{i-\frac{1}{2}, j+1})h_x^2-h_x^2h_y^2f_{i-\frac{1}{2}, j}}{2h_y^2+2h_x^2},
$$
$v$的未知量个数为$(N_x+2)\times(N_y+1)$，包含$x$方向左右两边的虚拟点。对**正则内点**为$i\in[1,N_x],j\in[1,N_y-1]$，**标准的五点差分格式(standard 5-point finite difference stencil)**，区域的四条边需要单独处理，四个角点归入左右边界。

在处理边界条件时，上下边界的角点需要知道左右边界上的未知量，而左右边界不需要知道上下边界上的未知量，因此先要遍历左右边界，再便利上下边界。

1. 左边界：$v_{0,j}$，$j\in[0,N_y]$

   **Nevmann边界条件**：将$\frac{\partial v}{\partial y}=g$进行有限差分方法离散，得到$\frac{v_{\frac{1}{2},j}-v_{-\frac{1}{2},j}}{h_x}=g_{0,j}$，即
   $$
   v_{-\frac{1}{2},j}=v_{\frac{1}{2},j}-h_xg_{0,j}
   $$
   **Dirichlet边界条件**：$\frac{v_{\frac{1}{2},j}+v_{-\frac{1}{2},j}}{2}=g_{0,j}$，即
   $$
   v_{-\frac{1}{2},j}=2g_{0,j}-v_{\frac{1}{2},j}
   $$

2. 右边界：$v_{N_x,j}$，$j\in[0,N_y]$

   **Nevmann边界条件**：将$\frac{\partial v}{\partial y}=g$进行有限差分方法离散，得到$\frac{v_{N_x+\frac{1}{2},j}-v_{N_x-\frac{1}{2},j}}{h_x}=g_{N_x,j}$，即
   $$
   v_{-\frac{1}{2},j}=v_{\frac{1}{2},j}-h_xg_{0,j}
   $$
   **Dirichlet边界条件**：$v_{N_x+\frac{1}{2},j}+v_{N_x-\frac{1}{2},j}=g_{N_x,j}$，即
   $$
   v_{N_x+\frac{1}{2},j}=g_{0,j}-v_{N_x-\frac{1}{2},j}
   $$

3. 上边界：$v_{i,0}$，$i\in[1,N_x-1]$

   **Nevmann边界条件**：将$\frac{\partial v}{\partial y}=g$进行中心差分离散，可得$\frac{v_{i,1}-v_{i,-1}}{2h_y}=g_{i,0}$ , 即

$$
v_\text{ghost}=v_{i,-1}=v_{i,1}-2h_yg_{i,0}
$$

$$
v_{i, 0}=\frac{(v_{i-1, 0}+v_{i+1, 0})h_y^2+(v_\text{ghost}+v_{i, 1})h_x^2-h_x^2h_y^2f_{i, 0}}{2h_y^2+2h_x^2}
$$

​       **Dirichlet边界条件**：$v_{i,0}={g}_{i,0}$

4. 下边界：$v_{i,Ny}$，$i\in[1,N_x-1]$

   **Nevmann边界条件**：类似可得$\frac{v_{i,N_y+1}-v_{i,N_y-1}}{h_y}=g_{i,N_y}$, 即

$$
v_{i,N_y+1}=v_{i,N_y-1}+h_yg_{i,N_y}
$$

$$
v_{i, N_y}=\frac{(v_{i-1, N_y}+v_{i+1, N_y})h_y^2+(v_{i, N_y-1}+v_{i, N_y+1})h_x^2-h_x^2h_y^2f_{i, N_y}}{2h_y^2+2h_x^2}
$$

​       **Dirichlet边界条件**：$v_{i,N_y}={g}_{i,N_y}$





$$

$$











## 使用投影方法对NS方程进行离散（连续方程带有源项）

### 控制方程

$$
\rho \left(\frac{\partial \mathbf{u}}{\partial t}+\mathbf{u}\nabla\mathbf{u}\right)=\mu\Delta \mathbf{u}-\nabla p+\mathbf{f}\\
\nabla\cdot \mathbf{u}={s}\tag{1}
$$

### 时间离散

$$
\rho \left(\frac{\mathbf{u}^{n+1}-\mathbf{u}^{n}}{\partial t}+\mathbf{u}^{n}\cdot\nabla\mathbf{u}^{n}\right)=\mu\Delta \mathbf{u}^{n+1}-\nabla p^{n+1}+\mathbf{f}^{n+1}\\
\nabla\cdot \mathbf{u}^{n+1}={s}^{n+1}\tag{2}
$$

### 投影方法

$$
\rho \left(\frac{\mathbf{u}^{*}-\mathbf{u}^{n}}{\Delta t}+\mathbf{u}^{n}\cdot\nabla\mathbf{u}^{n}\right)=\mu\Delta \mathbf{u}^{*}+\mathbf{f}^{n+1}\\\tag{3}
$$

$$
\rho\left(\frac{\mathbf{u}^{n+1}-\mathbf{u}^*}{\Delta t}\right)=-\nabla p^{n+1}\tag{4}
$$

对方程(4)左右两边**求散度**，再将方程组(2)中的二式代入，可得
$$
\rho\left(\frac{\nabla\cdot\mathbf{u}^{n+1}-\nabla\cdot\mathbf{u}^*}{\Delta t}\right)=-\Delta p^{n+1}\tag{5}
$$

$$
\rho\left(\frac{s^{n+1}-\nabla\cdot\mathbf{u}^*}{\Delta t}\right)=-\Delta p^{n+1}\tag{6}
$$

整理可得下面三个方程
$$
\frac{1}{\Delta t}\mathbf{u}^{*}-\frac{\mu}{\rho}\Delta \mathbf{u}^{*}=\frac{1}{\Delta t}\mathbf{u}^{n}+\frac{1}{\rho}\mathbf{f}^{n+1}-\mathbf{u}^{n}\cdot\nabla\mathbf{u}^{n}\tag{7}
$$

$$
\Delta p^{n+1}=\rho\left(\frac{\nabla\cdot\mathbf{u}^*-s^{n+1}}{\Delta t}\right)\tag{8}
$$

$$
{\mathbf{u}^{n+1}=\mathbf{u}^*}-\frac{\Delta t}{\rho}\nabla p^{n+1}\tag{9}
$$





### 空间离散

$$
bu_{i} =\frac{1}{\Delta t}\mathbf{u}^{n}_i+\frac{1}{\rho}\mathbf{f}^{n+1}_i
$$

$$
A\mathbf{u}^*=\frac{1}{\Delta t}\mathbf{u}^{*}_i-\frac{\mu}{\rho h^2}\left(\mathbf{u}^{*}_{i-1}-2\mathbf{u}^{*}_{i}+\mathbf{u}^{*}_{i+1}\right)
$$

$$
bp_{i}=\frac{\rho}{\Delta t}\left(\frac{\mathbf{u}_i^*-\mathbf{u}_{i-1}^*}{h}-s^{n+1}\right)
$$


$$
\mathbf{u}^{n+1}_i=\mathbf{u}^*_i-\frac{\Delta t}{\rho h}(p^{n+1}_{i}-p^{n+1}_{i-1})
$$








$$
\begin{aligned}
& \rho\left(\frac{\partial u}{\partial t}+(\vec{u} \cdot \nabla) u\right)=-\frac{\partial p}{\partial x}+\mu \nabla^2 u+f_1 \\
& \rho\left(\frac{\partial v}{\partial t}+(\vec{u} \cdot \nabla) v\right)=-\frac{\partial p}{\partial y}+\mu \nabla^2 v+f_2.
\end{aligned}
$$

$$
\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0
$$


$$
\begin{align*}
&\rho\left(\frac{\partial u_{i-\frac{1}{2}, j}}{\partial t}+\mathbf{u}_{i-\frac{1}{2}, j} \cdot \nabla u_{i-\frac{1}{2}, j}\right)&\\
=&-\frac{p_{i j}-p_{i-1, j}}{h_x}+\mu \frac{u_{i+\frac{1}{2}, j}-2 u_{i-\frac{1}{2}, j}+u_{i-\frac{3}{2}, j}}{h_x^2}\\
&+\mu \frac{u_{i-\frac{1}{2}, j+1}-2 u_{i-\frac{1}{2}, j}+u_{i-\frac{1}{2},j-1}}{h_y^2}+(f_1)_{i-\frac{1}{2}, j}
\end{align*}
$$

$$
\begin{align*}
&\rho\left(\frac{\partial v_{i, j-\frac{1}{2}}}{\partial t}+\mathbf{u}_{i, j-\frac{1}{2}} \cdot \nabla v_{i, j-\frac{1}{2}}\right)\\
=&-\frac{p_{i j}-p_{i, j-1}}{h_y}+\mu \frac{v_{i+1,j-\frac{1}{2}}-2 v_{i,j-\frac{1}{2}}+v_{i-1, j-\frac{1}{2}}}{h_x^2}\\
+&\mu \frac{v_{i,j+\frac{1}{2}}-2 v_{i, j-\frac{1}{2}}^2+v_{i,j-\frac{3}{2}}}{h_y^2}+(f_2)_{i, j-\frac{1}{2}}
\end{align*}
$$

$$
\frac{u_{i+\frac{1}{2},j}-u_{i-\frac{1}{2},j}}{h_x}+\frac{v_{i,j+\frac{1}{2}}-v_{i,j-\frac{1}{2}}}{h_y}=0
$$

摘自Newren的文章，迎风格式：

$$
\begin{aligned}
& H(x)= \begin{cases}1, & x>0 \\
0, & \text { otherwise }\end{cases} \\
& (\mathbf{u} \cdot \nabla \mathbf{u})_{i j}^n=\left[\begin{array}{l}
H\left(u_{i j}^n\right) u_{i j}^n \frac{u_{i+1, j}^n-u_{i j}^n}{h}+H\left(-u_{i j}^n\right) u_{i j}^n \frac{u_{i j}^n-u_{i-1, j}^n}{h}+H\left(v_{i j}^n\right) v_{i j}^n \frac{u_{i j, 1+1}^n-u_{i j}^n}{h}+H\left(-v_{i j}^n\right) v_{i j}^n \frac{u_{i j}^n-u_{i j}^n}{h} \\
H\left(u_{i j}^n\right) u_{i j}^n \frac{v_{i+1, j}^n-v_{i j}^n}{h}+H\left(-u_{i j}^n\right) u_{i j}^n \frac{v_{i j}^n-v_{i-1, j}^n}{h}+H\left(v_{i j}^n\right) v_{i j}^n \frac{v_{i, j+1}^n-v_{i j}^n}{h}+H\left(-v_{i j}^n\right) v_{i j}^n \frac{v_{i j}^n-v_{i, j-1}^n}{h}
\end{array}\right] . \\
&
\end{aligned}
$$





# Helmholtz方程

$$
\frac{1}{\Delta t}u+\frac{\mu}{\rho}\Delta u=f
$$

投影方法中，动量方程的空间离散

$$
\frac{1}{\Delta t}u_{i, j-\frac{1}{2}}^*-\frac{\mu}{\rho}\left(
\frac{u_{i-1, j-\frac{1}{2} }^*-2u_{i, j-\frac{1}{2}}^*+u*_{i+1, j-\frac{1}{2}}^*}{h_x^2} +\frac{u_{i, j-\frac{3}{2}}{-2 u_{i, j-\frac{1}{2}}^*}^*+u_{i, j+\frac{1}{2}}^*}{h_y^2}\right)=(b_1)_{i, j-\frac{1}{2}}
$$

$$
\left[\frac{1}{\Delta t}+\frac{2 \mu}{\rho}\left(\frac{1}{h_x^2}+\frac{1}{h_y^2}\right)\right] u_{i, j-\frac{1}{2}}^*=\left(b_1\right)_{i, j-\frac{1}{2}}+\frac{\mu}{\rho}\left(\frac{u_{i-1,j-\frac{1}{2}}^*+u_{i+1, j-\frac{1}{2}}^*}{h_x^2}
+\frac{u_{i, j-\frac{3}{2}}^*+u_{i,j+\frac{1}{2}}^*}{h_y^2}\right)
$$


$$
\frac{1}{\Delta t} v_{i-\frac{1}{2}, j}^*-\frac{\mu}{\rho}\left(\frac{v_{i-\frac{3}{2}, j}^*-2 v_{i-\frac{1}{2}, j}^*+v_{i+\frac{1}{2}, j}^*}{h_x^2}
+\frac{v_{i-\frac{1}{2},j-1}^*-2 v_{i-\frac{1}{2}, j}^*+v_{i-\frac{1}{2}, j+1}^*}{h_y^2}\right)=\left(b_2\right)_{i-\frac{1}{2}, j}
$$

$$
{\left[\frac{1}{\Delta t}+\frac{2 \mu}{\rho}\left(\frac{1}{h_x^2}+\frac{1}{h y^2}\right)\right] v_{i-\frac{1}{2}, j}^*=\left(b_2\right)_{i-\frac{1}{2}, j}+} 
\frac{\mu}{\rho}\left(\frac{v_{i-\frac{3}{2}, j}^*+v_{i+\frac{1}{2}, j}^*}{h_x^2}+\frac{v_{i-\frac{1}{2},j-1}^*+v_{i-\frac{1}{2}, j+1}^*}{h_y^2}\right)
$$

