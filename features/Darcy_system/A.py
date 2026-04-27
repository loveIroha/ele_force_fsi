from fenics import *
from mshr import *
mesh_path = '.'
# generate a circle mesh 
_domain = Circle(Point(0.6,0.5),0.2)
_mesh = generate_mesh(_domain,20)
mesh_file = XDMFFile(mesh_path+"circle_20.xdmf")
mesh_file.write(_mesh)
mesh_file.close()
print("10\nhmin : ", _mesh.hmin(), "hmax : ", _mesh.hmax() )