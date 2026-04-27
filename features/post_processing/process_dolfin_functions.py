from dolfin import *

# 1. 定义函数空间
nx = ny = 128
mesh = UnitSquareMesh(nx, ny)
Q = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 2)
W = VectorFunctionSpace(mesh, 'P', 1)

# 2. 定义函数
q = Expression('1', degree=2)
v = Expression(('1','2'), degree=2)
q = project(q,Q)
v = project(v, V)
w = project(v, W)
q.rename("q", "")
v.rename("v", "")
w.rename("w", "")

# 3. 输出函数(可重新读入)
f_out = XDMFFile(mesh.mpi_comm(), "test.xdmf")
f_out.parameters['rewrite_function_mesh'] = False
f_out.parameters["functions_share_mesh"] = True
f_out.parameters["flush_output"] = True
f_out.write_checkpoint(q, "q", 0.3, XDMFFile.Encoding.HDF5, True)
f_out.write_checkpoint(v, "v", 0.2, XDMFFile.Encoding.HDF5, True)
f_out.write_checkpoint(w, "w", 0.1, XDMFFile.Encoding.HDF5, True)
f_out.close()

# 输出函数(不可重新读入)
f_out_view = XDMFFile(mesh.mpi_comm(), "test_view.xdmf")
f_out_view.parameters['rewrite_function_mesh'] = False
f_out_view.parameters["functions_share_mesh"] = True
f_out_view.parameters["flush_output"] = True
f_out_view.write(q,0.5,XDMFFile.Encoding.HDF5)

# 输出网格
f_out_mesh = XDMFFile(mesh.mpi_comm(), "mesh.xdmf")
f_out_mesh.write(mesh)
f_out_mesh.close()

# 读入网格
mesh_in = Mesh()
f_mesh_in = XDMFFile("mesh.xdmf")
f_mesh_in.read(mesh_in)
f_mesh_in.close()

# 读入函数
Q = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 2)
W = VectorFunctionSpace(mesh, 'P', 1)
q = Function(Q)
v = Function(V)
W = Function(W)

f_in = XDMFFile(mesh.mpi_comm(), "test.xdmf")
f_in.read_checkpoint(q,"q",0)
f_in.read_checkpoint(v,"v",0)
f_in.read_checkpoint(w,"w",0)
