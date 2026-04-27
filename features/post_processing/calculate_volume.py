import numpy
from scipy.spatial import ConvexHull
magic = numpy.fromfile("./endocardium_points_0.734323",dtype='float64')
magic.resize((magic.size//3,3))
volume = ConvexHull(magic).volume







