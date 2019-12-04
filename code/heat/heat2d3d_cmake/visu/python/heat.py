#
# use Mayavi2/Mlab to visualize 2D heat equation solution
# 
# P. Kestener (18-dec-2009)
#

#import scipy
import sys
from enthought.mayavi import mlab 
import numpy as np

for i in sys.argv:
    print i


if (len(sys.argv)>3):
    # read xsm data
    filename=sys.argv[3]
    f=open(filename)
    header = f.readline()
    nx,ny = header.split()[2].split('x')
    nx = int(nx)
    ny = int(ny)
    print "nx : "+str(nx)
    print "ny : "+str(ny)
    read_data = np.fromfile(file=f, dtype=np.float32).reshape((nx,ny))
    f.close()

    # 3D visualization of data:
    #[x,y]=np.mgrid[0:1.0:1.0/nx,0:1.0:1.0/ny]
    mlab.figure(size=(400, 300))
    mlab.surf(read_data, warp_scale='auto')
    mlab.show()
else:
    print "You must provide data filename"
    print "usage: mayavi2 -x heat.py filename"

