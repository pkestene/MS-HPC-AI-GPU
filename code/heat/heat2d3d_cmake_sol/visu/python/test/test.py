import numpy
from numpy import sqrt,abs,r_

#generate data
dx,dy = 1.8,1.05
V = lambda x,y: 0.3**2/200.*sqrt((x-1)**2+y**2)*sqrt((x+1)**2+y**2)/abs(y)
x = r_[-dx:dx:200j]
y = r_[-dy:dy:200j]

#generate surface
from enthought.tvtk.tools import mlab
s = mlab.SurfRegular(x, y, V)

#simpler use:
fig = mlab.figure()
fig.add(s)
