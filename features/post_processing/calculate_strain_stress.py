"从真实左心室网格和纤维中ducoment中读取纤维方向"


from fenics import *
# ********* Mesh and I/O ********* 
mesh_file = XDMFFile("/home/kokkos/geometry/ventricle/LV/LV_real/mesh_scale.xdmf")
mesh = Mesh()
mesh_file.read(mesh)
mesh_file.close()
# Code for C++ evaluation of conductivity
fiberdirection_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class Conductivity : public dolfin::Expression
{
public:

  // Create expression with 3 components
  Conductivity() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<double>> c00;
  std::shared_ptr<dolfin::MeshFunction<double>> c01;
  std::shared_ptr<dolfin::MeshFunction<double>> c11;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Conductivity, std::shared_ptr<Conductivity>, dolfin::Expression>
    (m, "Conductivity")
    .def(py::init<>())
    .def_readwrite("c00", &Conductivity::c00)
    .def_readwrite("c01", &Conductivity::c01)
    .def_readwrite("c11", &Conductivity::c11);
}

"""


# Define conductivity components as MeshFunctions
c00 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/fibers_0.xml")
c01 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/fibers_1.xml")
c11 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/fibers_2.xml")
f0 = CompiledExpression(compile_cpp_code(fiberdirection_code).Conductivity(),
                       c00=c00, c01=c01, c11=c11, degree=0)


# Define conductivity components as MeshFunctions
c00 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/sheets_0.xml")
c01 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/sheets_1.xml")
c11 = MeshFunction("double", mesh, "/home/kokkos/geometry/ventricle/LV/LV_real/fiber/sheets_2.xml")
s0 = CompiledExpression(compile_cpp_code(fiberdirection_code).Conductivity(),
                       c00=c00, c01=c01, c11=c11, degree=0)



# 读入位移函数
W = VectorFunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'P', 1)
X = Function(W)
X.set_allow_extrapolation(True)
f_in = XDMFFile("/home/kokkos/ssh/npuheart/build_lv_old/solid/process.xdmf")
f_in.read_checkpoint(X,"position")


# 计算应变
F = variable(grad(X))
J = det(F)
C = F.T*F

strain = project(inner(f0, C*f0), Q)
File("strain.pvd") << strain

# 计算应力
a = 2400
b = 5.08
a_f = 14600
b_f = 4.15
a_s = 8700
b_s = 1.6
a_fs = 3000
b_fs = 1.3

F = variable(grad(X)) 
J = det(F)
C = F.T*F
I1 = tr(C)
I3 = det(C)

from ufl import max_value
I_4f = max_value(inner(f0, C*f0), 1.0)
I_4s = max_value(inner(s0, C*s0), 1.0)
I_8fs = inner(f0, C*s0)

W = a/2.0/b*exp(b*(I1-3))
W += a_f/2.0/b_f*(exp(b_f*(I_4f-1.0)*(I_4f-1.0))-1.0)
W += a_s/2.0/b_s*(exp(b_s*(I_4s-1.0)*(I_4s-1.0))-1.0)
W += a_fs/2.0/b_fs*(exp(b_fs*I_8fs*I_8fs)-1.0)
P = diff(W, F)

stress = project(inner(f0, P*f0), Q)
File("stress.pvd") << stress



