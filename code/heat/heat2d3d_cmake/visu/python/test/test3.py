from enthought.tvtk.tools import mlab
from scipy import *
def f(x,y):
    return sin(x+y) + sin(2*x-y) + cos(3*x+4*y)

x = linspace(-5, 5, 200)
y = linspace(-5, 5, 200)
fig = mlab.figure()
s = mlab.SurfRegularC (x,y,f)
fig.add( s )
fig.pop()
x = linspace( -5, 5, 100)
y = linspace( -5, 5, 100)
s = mlab.SurfRegularC (x,y,f)
fig.add(s)
