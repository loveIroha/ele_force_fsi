#include <AlgebraSolver/ConjugateGradient.h>
#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/BackgroundMesh2.h>
#include <MeshTools/ImmersedMesh.h>
#include <dolfin.h>

#include "Poisson.h"

namespace dolfin {

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const { values[0] = sin(5 * x[0]); }
};

} // namespace dolfin
