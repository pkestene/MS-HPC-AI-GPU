=======================================
load VTI file sequence into paraview :
=======================================
paraview --data=heat2d_cpu_..vti

============================================
load HDF5/XDMF file sequence into paraview :
============================================
paraview --data=filename.xmf

=======================================
make a GIF animation : 
=======================================
convert -delay 30 -loop 0 heat2d_ref_*.pgm heat2d_ref.gif
convert -delay 60 -loop 0 heat2d_ref_*.png heat2d_ref.gif

============================================
make a MPEG animation : *.png to output.avi
============================================
mencoder mf://*.png -mf w=800:h=600:fps=6:type=png -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi 

========================
MPEG visualisation :
========================
totem ./output.avi
mplayer ./output.avi
