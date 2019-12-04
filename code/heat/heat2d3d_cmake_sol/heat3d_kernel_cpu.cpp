/**
 * \file heat3d_kernel_cpu.cpp
 * \brief Solve 3D heat equation using finite difference methods.
 *
 * Use FTCS explicit scheme. 
 *
 * \date 27-dec-2009.
 */

#include "param.h"

#include "heat3d_kernel_cpu.h"


////////////////////////////////////////////////////////////////////////////////
//! solve 3D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void heat3d_ftcs_cpu_order2( float* data, float* dataNext)
{
  // heat3d ftcs time step
  for( unsigned int k = 1; k < NZ-1; ++k)
    for( unsigned int j = 1; j < NY-1; ++j) 
      for( unsigned int i = 1; i < NX-1; ++i) {
	// using column-major order
	unsigned int index = (k*NY+j)*NX+i;
	dataNext[index] = R3*data[index] + 
	  R*(data[index-1]  + data[index+1] + 
	     data[index-NX] + data[index+NX] +
	     data[index-NX*NY] + data[index+NX*NY]);
      }
} // heat3d_ftcs_cpu_order2 

////////////////////////////////////////////////////////////////////////////////
//! solve 3D heat equation using explicit FTCS scheme (4th order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void heat3d_ftcs_cpu_order4( float* data, float* dataNext)
{
  // heat3d ftcs time step
  for( unsigned int k = 2; k < NZ-2; ++k)
    for( unsigned int j = 2; j < NY-2; ++j) 
      for( unsigned int i = 2; i < NX-2; ++i) {
	// using column-major order
	unsigned int index = (k*NY+j)*NX+i;
	dataNext[index] = S3*data[index] + 
	  S*(-data[index-2]+16*data[index-1]+16*data[index+1]-data[index+2]
	     -data[index-2*NX]+16*data[index-NX]+16*data[index+NX]-data[index+2*NX]
	     -data[index-2*NX*NY]+16*data[index-NX*NY]+16*data[index+NX*NY]-data[index+2*NX*NY]);
      }
} // heat3d_ftcs_cpu_order4

