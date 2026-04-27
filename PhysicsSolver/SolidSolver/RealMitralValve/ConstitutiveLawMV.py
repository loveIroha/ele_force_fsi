from ufl import *
# 公共参数
# c_1s = 5.0e6    # 500kPa
# kappa = 5.0e6   # 与IBAMR中的 kappa beta 含义相反
# beta = 1.0e8    # 固定直管

# 前瓣膜
C1_ant = 1.7370e5
af_ant = 3.14596e5
bf_ant = 56.2429

# 后瓣膜
C1_post = 1.02e5
af_post = 5.0e5
bf_post = 63.48

# 腱索
C1_chordae = 9.0e7
C1_chordae_systole = 9.0e7
# C1_chordae_systole = 5.4e6

# 标号
marker_tube = 5             # 直管
marker_commsiure = 5099     # 瓣膜环边缘？(暂无)
marker_disk = 6             # 瓣膜环
marker_anterior = 7         # 前瓣膜
marker_posterior = 8        # 后瓣膜
marker_chordae = 10         # 腱索
marker_papillary = 11       # 乳头肌
marker_valve_edge = 5099    # 瓣膜边缘

def PK1_leaflet(X, C1, af, bf):
    I = Identity(len(X))
    F = variable(grad(X)+I)
    C = F.T*F
    J = det(F)
    I1 = tr(C)
    I2 = 0.5*(tr(C)*tr(C)-tr(C*C))
    I3 = det(C)
    I1 = I1*pow(I3, -1.0/3.0)
    I_4f = max_value(inner(f0, C*f0), 1.0)
    W = C1*(I1-3.0)+0.5*af/bf*(exp(bf*(I_4f-1.0)*(I_4f-1.0))-1.0)
    P = diff(W, F)
    return P


def PK1_anterior(X):
    return PK1_leaflet(X, C1_ant, af_ant, bf_ant)


def PK1_posterior(X):
    return PK1_leaflet(X, C1_post, af_post, bf_post)


def PK1_chordae(X):
    C1 = conditional(le(t_current, t_start_closing),
                     C1_chordae, C1_chordae_systole)
    I = Identity(len(X))
    F = variable(grad(X)+I)
    C = F.T*F
    I1 = tr(C)
    I3 = det(C)
    I1 = I1*pow(I3, -1.0/3.0)
    W = C1*(I1-3.0)
    P = diff(W, F)
    return P


def PK1_papillary(X):
    P = PK1_chordae(X)
    return P


def PK1_incompressible(X):
    I = Identity(len(X))
    F = variable(grad(X)+I)
    C = F.T*F
    return kappa*ln(det(C))*inv(F).T


def penalty_tube(X):
    return beta*X


def penalty_disk(X):
    return beta*X

def penalty_papillary(X):
    # 当前时间 t_current 
    # 瓣膜开始闭合时间 t_start_closing
    # 瓣膜完全闭合时间 t_end_closing
    # 心脏收缩(乳头肌移动)时间 t_systole
    # 健康心脏： (0.0, 0.5, -0.5)
    # 心肌梗死： (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.0, 1.5), (0.0, 0.0, 2.0)
    move_Y = 0.0
    move_Z = 0.5
    t_local = max_value(t_current, 0.0)           # 限制 t_local >= 0
    t_local = min_value(t_current, t_start_closing)     # 限制 t_local <= t_systole(0.2)
    # 乳头肌需要通过惩罚项指定位移 
    X_penalty = as_vector((0.0, X[1], X[2])) - as_vector((0.0, move_Y, move_Z))*t_local/t_start_closing
    # TODO: 收缩期结束后，王英的程序是将乳头肌瞬间拉回去，不知道是否正确。
    # TODO: 收缩开始前乳头肌是固定的！
    # TODO: struct chord cap point 是什么区域，需要始终固定？
    return beta*X_penalty


element = VectorElement("Lagrange", tetrahedron, 1)
real = FiniteElement("Real", tetrahedron, 0)

N = FacetNormal(tetrahedron)            # 外法向量
U = TrialFunction(element)
V = TestFunction(element)

kappa = Coefficient(real)               # 体积惩罚系数
beta = Coefficient(real)                # 位移惩罚系数
t_current = Coefficient(real)           # 当前时间

t_systole = Coefficient(real)           # 收缩时间
t_end_closing = Coefficient(real)       # 收缩开始时间
t_start_closing = Coefficient(real)     # 收缩开始时间

X = Coefficient(element)
f0 = Coefficient(element)
X0 = SpatialCoordinate(tetrahedron)

# 弱形式
H = inner(U, V)*dx

H += inner(PK1_anterior(X),       grad(V)) * \
    dx(subdomain_id=marker_anterior,   degree=5)
H += inner(PK1_posterior(X),      grad(V)) * \
    dx(subdomain_id=marker_posterior,  degree=5)
H += inner(PK1_chordae(X),        grad(V)) * \
    dx(subdomain_id=marker_chordae,    degree=5)
H += inner(PK1_papillary(X),      grad(V)) * \
    dx(subdomain_id=marker_papillary,  degree=5)

# 不可压惩罚项
H += inner(PK1_incompressible(X), grad(V))*dx(degree=5)

# 惩罚位移
H += inner(penalty_disk(X), V)*dx(subdomain_id=marker_disk, degree=5)
H += inner(penalty_tube(X), V)*dx(subdomain_id=marker_tube, degree=5)
H += inner(penalty_papillary(X), V)*dx(subdomain_id=marker_papillary, degree=5)

# 分离左右手
a = lhs(H)
L = rhs(H)
