/**
 * \file heat_kernel_cpu.cpp
 * \brief Solve 2D / 3D heat equation using finite difference methods.
 *
 * Use FTCS explicit scheme. 
 *
 * \date Dec 17 2009.
 */

#include "param.h"

#include "heat_kernel_cpu.h"

////////////////////////////////////////////////////////////////////////////////
//! solve 2D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void
heat2d_ftcs_cpu_order2( const real_t* data, real_t* dataNext )
{
  real_t &R  = o2.R;
  real_t &R2 = o2.R2;

  // heat2d ftcs time step
#pragma omp parallel for
  for( unsigned int j = 1; j < NY-1; ++j) {
#pragma ivdep
#pragma omp simd
    for (unsigned int i = 1; i < NX - 1; ++i) {
      // using column-major order
      unsigned int index = j * NX + i;
      dataNext[index] =
          R2 * data[index] + R * (data[index - 1] + data[index + 1] +
                                  data[index - NX] + data[index + NX]);
    } // end for i
  } // end for j

} // heat2d_ftcs_cpu_order2

////////////////////////////////////////////////////////////////////////////////
//! solve 2D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void
heat2d_ftcs_cpu_order2b( const real_t* data, real_t* dataNext )
{
  real_t &R   = o2.R;
  real_t &R2b = o2.R2b;

  // heat2d ftcs time step
#pragma omp parallel for
  for( unsigned int j = 1; j < NY-1; ++j) { 
#pragma ivdep
#pragma omp simd
    for (unsigned int i = 1; i < NX - 1; ++i) {

      // using column-major order
      int index = j * NX + i;

      real_t tmp = R2b * data[index];

      tmp +=
          R * (data[index + 1 - NX] + data[index - NX] + data[index - 1 - NX] +
               data[index + 1     ] + data[index     ] + data[index - 1     ] +
               data[index + 1 + NX] + data[index + NX] + data[index - 1 + NX]);

      dataNext[index] = tmp;
    } // end for i
  } // end for j

} // heat2d_ftcs_cpu_order2b

////////////////////////////////////////////////////////////////////////////////
//! solve 2D heat equation using explicit FTCS scheme (4th order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void
heat2d_ftcs_cpu_order4( const real_t* data, real_t* dataNext )
{
  real_t &S  = o4.S;
  real_t &S2 = o4.S2;

  // heat2d ftcs time step
#pragma omp parallel for
  for( unsigned int j = 2; j < NY-2; ++j) { 
#pragma ivdep
#pragma omp simd
    for (unsigned int i = 2; i < NX - 2; ++i) {
      // using column-major order
      unsigned int index = j * NX + i;
      dataNext[index] =
          S2 * data[index] +
          S * (-data[index - 2     ] + 16 * data[index - 1]  + 16 * data[index + 1] - data[index + 2] -
                data[index - 2 * NX] + 16 * data[index - NX] + 16 * data[index + NX] - data[index + 2 * NX]);
    } // end for i
  } // end for j

} // heat2d_ftcs_cpu_order4


////////////////////////////////////////////////////////////////////////////////
//! solve 2D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void
heat2d_ftcs_cpu_order2_with_mask( const real_t* data, 
				  real_t* dataNext, 
				  const int* mask)
{
  real_t &R  = o2.R;
  real_t &R2 = o2.R2;

  // heat2d ftcs time step
#pragma omp parallel for
  for( unsigned int j = 1; j < NY-1; ++j) {
#pragma ivdep
#pragma omp simd
    for (unsigned int i = 1; i < NX - 1; ++i) {
      // using column-major order
      unsigned int index = j * NX + i;
      real_t tmp = R2 * data[index] + R * (data[index - 1 ] + data[index + 1 ] +
                                           data[index - NX] + data[index + NX]);
      dataNext[index] = tmp * mask[index] + 1 - mask[index];
    } // end for i
  } // end for j

} // heat2d_ftcs_cpu_order2_with_mask

////////////////////////////////////////////////////////////////////////////////
//! solve 3D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void heat3d_ftcs_cpu_order2( const real_t* data, real_t* dataNext )
{
  real_t &R  = o2.R;
  real_t &R3 = o2.R3;

  // heat3d ftcs time step
#pragma omp parallel for
  for( unsigned int k = 1; k < NZ-1; ++k) {
    for( unsigned int j = 1; j < NY-1; ++j) { 
#pragma ivdep
#pragma omp simd
      for (unsigned int i = 1; i < NX - 1; ++i) {
        // using column-major order
        unsigned int index = (k * NY + j) * NX + i;
        real_t tmp = R3 * data[index] +
                     R * (data[index - 1] + data[index + 1] + data[index - NX] +
                          data[index + NX] + data[index - NX * NY] +
                          data[index + NX * NY]);
        dataNext[index] = tmp;
      } // end for i
    } // end for j
  } // end for k

} // heat3d_ftcs_cpu_order2 

////////////////////////////////////////////////////////////////////////////////
//! solve 3D heat equation using explicit FTCS scheme (2nd order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void heat3d_ftcs_cpu_order2b( const real_t* data, real_t* dataNext )
{
  real_t &R  = o2.R;
  real_t &R3 = o2.R3b;

  // heat3d ftcs time step
#pragma omp parallel for
  for( unsigned int k = 1; k < NZ-1; ++k) {
    for( unsigned int j = 1; j < NY-1; ++j) { 
#pragma ivdep
#pragma omp simd
      for( unsigned int i = 1; i < NX-1; ++i) {
	// using column-major order
	unsigned int index = (k*NY+j)*NX+i;
	real_t tmp = R3*data[index];
	// 3x3 stencil
	for (int kk=-1; kk<2; kk++) {
	  for (int jj=-1; jj<2; jj++) {
            for (int ii = -1; ii < 2; ii++) {
              int index2 = index + ii + NX * jj + NX * NY * kk;
              tmp += R * data[index2];
            }
          }
        }
        dataNext[index] = tmp;
      } // end for i 
    } // end for j
  } // end for k

} // heat3d_ftcs_cpu_order2b 

////////////////////////////////////////////////////////////////////////////////
//! solve 3D heat equation using explicit FTCS scheme (4th order accurate)
//! 
//! @param data       data array, preallocated, contains initial data
//! @param dataNext   data array, preallocated, next time step data
////////////////////////////////////////////////////////////////////////////////
void heat3d_ftcs_cpu_order4( const real_t* data, real_t* dataNext )
{
  real_t &S  = o4.S;
  real_t &S3 = o4.S3;

  // heat3d ftcs time step
#pragma omp parallel for
  for( unsigned int k = 2; k < NZ-2; ++k) {
    for( unsigned int j = 2; j < NY-2; ++j) { 
#pragma ivdep
#pragma omp simd
      for (unsigned int i = 2; i < NX - 2; ++i) {
        // using column-major order
        unsigned int index = (k * NY + j) * NX + i;
        dataNext[index] =
            S3 * data[index] +
            S * (-data[index - 2     ] + 16 * data[index - 1] + 16 * data[index + 1] - data[index + 2] 
                 -data[index - 2 * NX] + 16 * data[index - NX] + 16 * data[index + NX] - data[index + 2 * NX] 
                 - data[index - 2 * NX * NY] + 16 * data[index - NX * NY] + 16 * data[index + NX * NY] - data[index + 2 * NX * NY]);
      } // end for i
    } // end for j
  } // end for k

} // heat3d_ftcs_cpu_order4
