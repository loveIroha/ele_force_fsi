from ufl import *

# Boundary markers
markers = {"left": 5, "right": 3}
markers["left"]
def strain_energy(F):

    # Invariants
    J = det(F)
    C = F.T*F
    I1 = tr(C)
    I4 = inner(A, C*A)
    I5 = inner(A, C*C*A)

    # Strain energy
    W = G_T/2*(I1-3)
    W += (G_T-G_L)/2*(2*I4-I5-1)
    W += (E_L+G_T-4*G_L)/8*(I4-1)**2

    # stablization
    W += kappa/2*(ln(J))**2
    return W

def first_PK_stress(X):
    F = grad(X) + Identity(len(X))
    F = variable(F)
    PP = diff(strain_energy(F), F)
    return PP

element = VectorElement("Lagrange", tetrahedron, 1)
real = FiniteElement("Real", tetrahedron, 0)
pressure = Coefficient(real)

# 1. 是否施加不可压缩惩罚
# 2. 应变张量使用 pow(J, -float(2)/3) * F.T*F 还是 F.T*F
kappa = Coefficient(real) # Penalty for incompressibility , 1e5 by default
beta = Coefficient(real) # Penalty for fixed face, 1e5 by default

G_T = Coefficient(real)
G_L = Coefficient(real)
E_L = Coefficient(real)
A = Coefficient(element)
kappa = Coefficient(real)
    
U = TrialFunction(element)
V = TestFunction(element)
X = Coefficient(element)

# Out normal at reference configuration.
N = FacetNormal(tetrahedron)

# The deformation gradient
FF = grad(X) + Identity(len(X))

# The first Piola-Kirchhoff stress
PP = first_PK_stress(X)

F = inner(PP, grad(V))*dx(degree=5) + inner(U, V)*dx

# Pressure at the bottom face
N = as_vector([0,1,0])
F += pressure * inner(cofac(FF)*N,V)*ds(5)

# Penalty for fixed face
F += beta*inner(X,V)*ds(3)

a = lhs(F)
L = rhs(F)
