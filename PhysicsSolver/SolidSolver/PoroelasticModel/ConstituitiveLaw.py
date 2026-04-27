from fenics import *
from ufl import *
# Define elements
element = FiniteElement("Lagrange", tetrahedron, 1)
vector_element = VectorElement("Lagrange", tetrahedron, 1)
constant_element = FiniteElement("Real", tetrahedron, 0)


kappa_1 = 0
kappa_2 = 0
K_s = 0
rho = 0

X = Coefficient(vector_element)
F = variable(grad(X))
J = det(F)
C = F.T*F
I1 = tr(C)
I2 = 0.5*(tr(C)**2-tr(C*C))


M = Coefficient(element)

W_hyp = kappa_1*(I1-3)+kappa_2*(I2-3)
W_bulk = K_s*ln(J-M/rho)**2

Phi = 0

W = W_hyp + W_bulk + Phi

P = diff(W,F)



