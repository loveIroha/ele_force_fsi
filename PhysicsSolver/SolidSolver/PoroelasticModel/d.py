from fenics import *
from ufl import *

# Define parameters
# Density
rho = 1e3

# Source
S = 0

# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)

# Quantities
X = Coefficient(vector_element)
F = grad(X)
J = det(F)
M0 = Coefficient(element)
W = Coefficient(vector_element)

# time step
dt  = Coefficient(constant_element)

# Test function and trial function
M = TrialFunction(element)
V = TestFunction(element)

F = 2.0*(M-M0)/dt*V*dx
F -= rho*S*V*dx
F += rho*div(W)*V*dx

a = lhs(F)
L = rhs(F)


