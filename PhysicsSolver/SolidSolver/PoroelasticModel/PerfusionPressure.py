
# ### 灌注压力的求解
# - $M_b$:常数
# - $p$：是灌注压力，和流体的压力无关
# - $f(J)=\frac{2(J(\mathbf{X}, t)-1-\ln (J(\mathbf{X}, t)))}{(J(\mathbf{X}, t)-1)^2}$
# - E:一个标量，与$m^n$和$J^n$

# $$
# \frac{p^{n+\frac{1}{2}}-p^n}{\triangle t / 2}-M_b f\left(J^n\right) \nabla_{\mathbf{X}} \cdot\left(J^n \mathbb{F}^{-1} \mathbf{K} \mathbb{F}^{-T} \nabla_{\mathbf{X}} p^{n+\frac{1}{2}}\right)=E\left(m^n, J^n\right)
# $$

# from fenics import *
# from ufl import *

# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)

# Test function and trial function
P = TrialFunction(element)
V = TestFunction(element)

# Define parameters
# Biot modulus
Mb = 2.18e5

# Permeability tensor
K = 1e-7*Identity(3)

# Density
rho = 1e3

# Penalty coefficient
kappa = 0.01

# Parameter of the skeleton
b = 1.0

# Initial porosity
phi0 = 0.1

# Source
S = 0

# time step
dt  = Coefficient(constant_element)

# Deformation gradient
X  = Coefficient(vector_element)
F  = grad(X)
J  = det(F)
X_1  = Coefficient(vector_element)
F_1  = grad(X_1)
J_1  = det(F_1)

# Variables from last time step
P0 = Coefficient(element)
M  = Coefficient(element)
M_1 = Coefficient(element)

def f(J):
    return 2.0*(J-1.0-ln(J))/(J-1)**2

def A(J):
    return Mb*f(J)/rho

def B(M,J):
    return -Mb*f(J)*b+Mb*(2/(J*(J-1))-4*(J-1-ln(J))/(J-1)**3)*(b(1-J)+M/rho)

def C(M,M_1):
    return (kappa*rho/(M+rho*phi0)-kappa*rho/(M_1+rho*phi0))/dt
    
def E(M,M_1,J,J_1):
    return A(J)*rho*S + B(M,J)*(J-J_1)/dt + C(M,M_1)


# Define the variational form
AA = J*inv(F)*K*inv(F).T

# TODO: Zero Neumann boundary condition has not been considered
F = inner(2.0*(P-P0)/dt,V)*dx + Mb*f(J)*inner(AA*grad(P),grad(V))*dx-E(M,M_1,J,J_1)*V*dx

a = lhs(F)
L = rhs(F)

