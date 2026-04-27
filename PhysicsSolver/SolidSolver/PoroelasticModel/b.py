

from fenics import *
from ufl import *

# Permeability tensor
K = 1e-7*Identity(3)


# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)

X = Coefficient(vector_element)
F = grad(X)
J = det(F)

P = Coefficient(element)


# Define the variational form
AA = J*inv(F)*K*inv(F).T

# Test function and trial function
W = TrialFunction(vector_element)
V = TestFunction(vector_element)

F = inner(W,V)*dx + inner(AA*grad(P),V)*dx
a = lhs(F)
L = rhs(F)


