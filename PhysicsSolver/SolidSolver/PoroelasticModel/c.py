



from fenics import *
from ufl import *

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

# Inital Pressure
P0 = 0

# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)

# Quantities
X = Coefficient(vector_element)
F = grad(X)
J = det(F)
P = Coefficient(element)
M0 = Coefficient(element)



def f(J):
    return 2.0*(J-1.0-ln(J))/(J-1)**2

def g(P):
    return P - P0 + kappa*rho/(M0+rho*phi0)

# Test function and trial function
M = TrialFunction(element)
V = TestFunction(element)

F = M*V*dx - rho*(1.0/(Mb*f(J))*g(P))*V*dx - rho*b*(J-1)*V*dx

a = lhs(F)
L = rhs(F)


