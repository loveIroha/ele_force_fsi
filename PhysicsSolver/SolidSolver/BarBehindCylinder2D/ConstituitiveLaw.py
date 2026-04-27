element = VectorElement("Lagrange", triangle, 1)
real = FiniteElement("Real", triangle, 0)

# 区域标记
cylinder = 1
bar = 2

U = TrialFunction(element)
V = TestFunction(element)

# 本构参数
G_s = Coefficient(real)
nv = Coefficient(real)
K_s = 2*G_s*(1+nv)/(1-2*nv)

# 已知变量，例如位移
X = Coefficient(element)

# 形变梯度，应变张量等
FF = grad(X)+Identity(len(X))
CC = FF.T*FF
JJ = det(FF)
I1 = tr(CC)

# 第一PK应力张量(neo-Hookean本构)
PP_dev = G_s*pow(JJ,-2/3)*(FF-I1/3*inv(FF).T)

# 增加不可压条件 
PP_dil = K_s*JJ*ln(JJ)*inv(FF).T

# 弱形式
F = inner(U, V)*dx

# 尾巴区域为neo-Hookean本构
F += inner(PP_dev, grad(V))*dx(2)

# 尾巴区域都施加不可压条件
F += inner(PP_dil, grad(V))*dx(2)
 
# 双线性型及右端项
a = lhs(F)
L = rhs(F)
