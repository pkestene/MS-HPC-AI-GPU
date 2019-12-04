/**
 * \file output.h
 * \brief Routine to save output results into a file.
 *
 * \date 5-jan-2010
 */
#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "param.h" // get real_t typedef 

// for graphics output (PNG)
#ifdef USE_MGL
#include <mgl2/mgl.h>
#endif

void save_bin(real_t* data, const char* prefix, int time, int sizeX, int sizeY);
void save_pgm(real_t* data, const char* prefix, int time, int sizeX, int sizeY);
void save_mgl(real_t* data, const char* prefix, int time, int sizeX, int sizeY);


void save_bin_3d(real_t* data, const char* prefix, int time, 
		 int sizeX, int sizeY, int sizeZ);
void save_mgl_3d(real_t* data, const char* prefix, int time, 
		 int sizeX, int sizeY, int sizeZ);

/* for 2D or 3D */
void save_vtk(real_t* data, const char* prefix, int time);

#if USE_HDF5
#include <hdf5.h>
#endif
void save_hdf5(real_t* data, const char* prefix, int time, int compressionLevel=0);
void write_xdmf_wrapper(const char* prefix, int totalNumberOfSteps, int deltaT);


/** a simple routine to check for endianess at runtime */
inline bool isBigEndian()
{
  const int i = 1;
  return ( (*(char*)&i) == 0 );
}


#endif // OUTPUT_H_
