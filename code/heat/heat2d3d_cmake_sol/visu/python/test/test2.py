import scipy

# prepare some interesting function:
def f(x, y):
    return 3.0*scipy.sin(x*y+1e-4)/(x*y+1e-4)

x = scipy.arange(-7., 7.05, 0.1)
y = scipy.arange(-5., 5.05, 0.1)

# 3D visualization of f:
from enthought.tvtk.tools import mlab
fig = mlab.figure()
s = mlab.SurfRegular(x, y, f)
fig.add(s)
