#!/usr/bin/env python
import sys
from pylab import *
import numpy as np

# read binary/XSM data
if (len(sys.argv) > 1):
    f=open(sys.argv[1])
    header=f.readline()
    nx,ny = header.split()[2].split('x')
    nx = int(nx)
    ny = int(ny)
    data = np.fromfile(file=f, dtype=np.float32).reshape((ny,nx))
    f.close()
else:
    print "You must provide data filename."

dpi = rcParams['figure.dpi']
figsize = ny/dpi, nx/dpi

figure(figsize=figsize)
ax = axes([0,0,1,1], frameon=False)
ax.set_axis_off()
im = imshow(data, origin='lower')
#im = imshow(data, origin='lower', cmap=cm.gray)

show()
