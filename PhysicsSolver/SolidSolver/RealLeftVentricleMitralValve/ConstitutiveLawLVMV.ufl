from ufl import *
# 公共参数
C1 = 1.0e5   # 500kPa
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

# boundary markers
marker_vessel_inner = 4092  # 血管内表面
marker_aorta_down = 4090    # 主动脉
marker_aorta_up = 4093
marker_atrium_up = 4094     # 左心房
marker_atrium_down = 4091
marker_endocardium = 4097 # left ventricle endocardium
marker_epicardium = 4096  # left ventricle epicardium
marker_base= 4098         # left ventricle base

# material markers
marker_vessel = 2
marker_left_ventricle = 1
marker_outflow_tract = 3
marker_papillary = 11
marker_chordae = 10
marker_anterior = 7
marker_posterior = 8
marker_housing_disk = 6


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

def PK1_neo_hookean(X):
    I = Identity(len(X))
    F = variable(grad(X)+I)
    C = F.T*F
    I1 = tr(C)
    I3 = det(C)
    I1 = I1*pow(I3, -1.0/3.0)
    W = C1*(I1-3.0)
    P = diff(W, F)
    return P

def PK1_chordae(X):
    C1 = conditional(le(t_current, t_end_diastole),
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


def penalty_displacement(X):
    return beta*X

def penalty_papillary(X, move_X, move_Y, move_Z):
    # 当前时间 t_current 
    # 瓣膜开始闭合时间 t_end_diastole
    # 瓣膜完全闭合时间 t_end_closing
    # 心脏收缩(乳头肌移动)时间 t_systole
    # 健康心脏： (0.0, 0.5, -0.5)
    # 心肌梗死： (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.0, 1.5), (0.0, 0.0, 2.0)
    # TODO: 乳头肌在舒张期[0,0.185]移动，之后的收缩期[0.185,0.385]不再发生位移。事实上，收缩期需要位移。
    t_local = max_value(t_current, 0.0)           # 限制 t_local >= 0
    t_local = min_value(t_current, t_end_diastole)     # 限制 t_local <= 0.185
    # 乳头肌需要通过惩罚项指定位移 
    X_penalty = as_vector(
        (X[0], X[1], X[2])) - as_vector((move_X, move_Y, move_Z))*t_local/t_end_diastole
    # TODO: 收缩期结束后，王英的程序是将乳头肌瞬间拉回去，不知道是否正确。
    # TODO: 收缩开始前乳头肌是固定的！
    # TODO: struct chord cap point 是什么区域，需要始终固定？
    return 10.0*beta*X_penalty
    # return 10.0*beta*X



def press_endocardium(X):
    I = Identity(len(X))
    F = variable(grad(X)+I)
    return pressure * det(F)*inv(F)*N

element = VectorElement("Lagrange", tetrahedron, 1)
real = FiniteElement("Real", tetrahedron, 0)

kappa = Coefficient(real)                # 体积惩罚系数
beta = Coefficient(real)                 # 位移惩罚系数
t_current = Coefficient(real)            # 当前时间

t_systole = Coefficient(real)            # 收缩时间
t_end_closing = Coefficient(real)        # 收缩开始时间
t_end_diastole = Coefficient(real)      # 收缩开始时间

pressure = Coefficient(real)             # 表面施加压力

posterior_papillary_x = 0
posterior_papillary_y = 0
posterior_papillary_z = 0

anterior_papillary_x = 0
anterior_papillary_y = 0
anterior_papillary_z = 0

N = FacetNormal(tetrahedron)
U = TrialFunction(element)
V = TestFunction(element)

X = Coefficient(element)
f0 = Coefficient(element)
X0 = SpatialCoordinate(tetrahedron)

# 弱形式
H = inner(U, V)*dx

# # 不可压惩罚项
# H += inner(PK1_incompressible(X), grad(V))*dx(degree=5)

# 所有部分都是neo-hookean材料
H += inner(PK1_neo_hookean(X), grad(V))*dx(degree=5)

# # 固定出入口
# H += inner(penalty_displacement(X), V)*dx(subdomain_id=marker_outflow_tract, degree=5)

# # 固定前后乳头肌
# H += inner(penalty_papillary(X, anterior_papillary_x, anterior_papillary_y, anterior_papillary_z), V)*dx(subdomain_id=marker_anterior, degree=5)
# H += inner(penalty_papillary(X, posterior_papillary_x, posterior_papillary_y, posterior_papillary_z), V)*dx(subdomain_id=marker_posterior, degree=5)

# 表面施加压力
H += inner(press_endocardium(X), V)*ds(subdomain_id=marker_endocardium, degree=5)

# 分离左右手
a = lhs(H)
L = rhs(H)
