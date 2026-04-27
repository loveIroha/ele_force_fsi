from fenics import *
from ufl import *

# Define parameters
# Source
S = 0

# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)

# Quantities
W = Coefficient(vector_element)

# time step
dt  = Coefficient(constant_element)

# Test function and trial function
S_ = TrialFunction(element)
V = TestFunction(element)

F = S_*V*dx-S*V*dx+div(W)*V*dx

a = lhs(F)
L = rhs(F)


