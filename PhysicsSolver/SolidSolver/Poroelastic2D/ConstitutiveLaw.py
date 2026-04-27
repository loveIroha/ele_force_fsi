from fenics import *
from ufl import *
# Define elements
element = FiniteElement("Lagrange", triangle, 1)
vector_element = VectorElement("Lagrange", triangle, 1)
constant_element = FiniteElement("Real", triangle, 0)

# Define parameters
kappa_1 = 2e4
kappa_2 = 330
K_s = 2.2e6

# Biot modulus
Mb = 2.18e6

# Density
rho = 1.0

# Penalty coefficient
kappa = 0.1

# Parameter of the skeleton
b = 1.0

# Initial porosity
phi0 = 0.1

N = FacetNormal(triangle)
X = Coefficient(vector_element)
F = variable(grad(X)+Identity(2))
J = det(F)
C = F.T*F
I1 = tr(C)
I2 = 0.5*(tr(C)**2-tr(C*C))

def f(J):
    f1 = 1
    f2 = 2.0*(J-1.0-ln(J))/((J-1)**2)
    return conditional(le(abs(J-1),1e-6),f1,f2)

M= Coefficient(element)

W_hyp = kappa_1*(I1-2)+kappa_2*(I2-2)
W_bulk = K_s*ln(J-M/rho)**2
Phi = -Mb*b*M/rho*(J - 1)*f(J) + 0.5*Mb*(M/rho)**2*f(J) - kappa*ln(M/rho + phi0)
W = W_hyp + W_bulk + Phi
P = diff(W,F) 

#Define the variational problem
G = TrialFunction(vector_element)
V = TestFunction(vector_element) 
F = inner(P,grad(V))*dx + inner(G,V)*dx

# Constraint left and bottom edge
F += 1e4*inner(X,V)*ds(subdomain_id=1)
F += 1e4*inner(X,V)*ds(subdomain_id=2)

a = lhs(F)
L = rhs(F)