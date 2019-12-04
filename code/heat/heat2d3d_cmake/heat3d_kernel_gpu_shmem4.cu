/**
 * \file heat3d_kernel_gpu_shmem4.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * \author Pierre Kestener
 * \date 19-nov-2012
 */

#ifndef _HEAT3D_KERNEL_GPU_SHMEM4_H_
#define _HEAT3D_KERNEL_GPU_SHMEM4_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX 24
#define BLOCK_DIMY 16
#define BLOCK_INNER_DIMX (BLOCK_DIMX-2)
#define BLOCK_INNER_DIMY (BLOCK_DIMY-2)

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
__global__ void heat3d_ftcs_sharedmem4_order2_kernel(real_t* A, real_t* B, 
						     int pitch, 
						     int imax, int jmax, int kmax)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index (avoid external ghost cells)
  const int i = __mul24(bx, BLOCK_INNER_DIMX) + tx + 1;
  const int j = __mul24(by, BLOCK_INNER_DIMY) + ty + 1;
  
  const int index = pitch*j + i;
  const int dy=pitch;
  const int dz=pitch*jmax;

  // z-plane indexes for temperature at time t
  int z0, z1, z2, z3, z4, tmp;
  z0=0; z1=1; z2=2; z3=3; z4=4;
  
  // z-plane indexes for temperature at time t+1
  int zz1, zz2, zz3;
  zz1=0; zz2=1; zz3=2;
    
  // shmem0: temperature at t   : 2 ghost cells per dimension (1 left + 1 right)
  __shared__ real_t shmem0[5][BLOCK_DIMY+2][BLOCK_DIMX+2];
  // shmem1: temperature at t+1 : ghost cell not needed, but use them anyway
  // so that we can have the same indexes
  __shared__ real_t shmem1[3][BLOCK_DIMY+2][BLOCK_DIMX+2];

  /*
   * copy data from global mem to shared mem 
   */
  // initialize temperature at t (just copy from external memory)
  if( i<imax and j<jmax )
    {
      
      // fill bulk data
      shmem0[z0][ty+1][tx+1] = A[index+z0*dz];
      shmem0[z1][ty+1][tx+1] = A[index+z1*dz];
      shmem0[z2][ty+1][tx+1] = A[index+z2*dz];
      shmem0[z3][ty+1][tx+1] = A[index+z3*dz];
      shmem0[z4][ty+1][tx+1] = A[index+z4*dz];

  
      // fill shared memory ghost cells (left and right)
      if (tx == 0) // load ghost cells
	{
	  shmem0[z0][ty+1][tx] = A[index-1+z0*dz]; // x-1
	  shmem0[z1][ty+1][tx] = A[index-1+z1*dz];
	  shmem0[z2][ty+1][tx] = A[index-1+z2*dz];
	  shmem0[z3][ty+1][tx] = A[index-1+z3*dz];
	  shmem0[z4][ty+1][tx] = A[index-1+z4*dz];

	  if (i+BLOCK_DIMX < imax) {               // x+BLOCK_DIMX 
	    shmem0[z0][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+z0*dz]; 
	    shmem0[z1][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+z1*dz];
	    shmem0[z2][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+z2*dz];
	    shmem0[z3][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+z3*dz];
	    shmem0[z4][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+z4*dz];
	  }
	}
      
      // fill shared memory ghost cells (up and down)
      if (ty == 0)
	{
	  shmem0[z0][ty][tx+1] = A[index-1*dy+z0*dz];// y-1
	  shmem0[z1][ty][tx+1] = A[index-1*dy+z1*dz];// y-1
	  shmem0[z2][ty][tx+1] = A[index-1*dy+z2*dz];// y-1
	  shmem0[z3][ty][tx+1] = A[index-1*dy+z3*dz];// y-1
	  shmem0[z4][ty][tx+1] = A[index-1*dy+z4*dz];// y-1

	  if (j+BLOCK_DIMY < jmax) { // y+BLOCK_DIMY
	    shmem0[z0][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*dy+z0*dz];  
	    shmem0[z1][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*dy+z1*dz];
	    shmem0[z2][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*dy+z2*dz];
	    shmem0[z3][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*dy+z3*dz];
	    shmem0[z4][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*dy+z4*dz];
	  }
	}
    }
  // wait for all threads in the block to finish loading data
  __syncthreads();

  // initialize temperature at time t+1
  if( i>0      and j>0      and 
      i<imax-1 and j<jmax-1 ) {

    shmem1[zz1][ty+1][tx+1] = // z1 plane
      o2Gpu.R3 *  shmem0[z1][ty+1][tx+1] +
      o2Gpu.R  * (shmem0[z1][ty+1][tx  ] +
		  shmem0[z1][ty+1][tx+2] +
		  shmem0[z1][ty  ][tx+1] +
		  shmem0[z1][ty+2][tx+1] +
		  shmem0[z0][ty+1][tx+1] +
		  shmem0[z2][ty+1][tx+1] 
		  );
    shmem1[zz2][ty+1][tx+1] = // z2 plane
      o2Gpu.R3 *  shmem0[z2][ty+1][tx+1] +
      o2Gpu.R  * (shmem0[z2][ty+1][tx  ] +
		  shmem0[z2][ty+1][tx+2] +
		  shmem0[z2][ty  ][tx+1] +
		  shmem0[z2][ty+2][tx+1] +
		  shmem0[z1][ty+1][tx+1] +
		  shmem0[z3][ty+1][tx+1] 
		  );
    shmem1[zz3][ty+1][tx+1] = // z3 plane
      o2Gpu.R3 *  shmem0[z3][ty+1][tx+1] +
      o2Gpu.R  * (shmem0[z3][ty+1][tx  ] +
		  shmem0[z3][ty+1][tx+2] +
		  shmem0[z3][ty  ][tx+1] +
		  shmem0[z3][ty+2][tx+1] +
		  shmem0[z2][ty+1][tx+1] +
		  shmem0[z4][ty+1][tx+1] 
		  );
  }
 
  
  // do FTCS time steps update and copy back results to global mem buffer B
  for (int k=2; k<kmax-3; ++k) {
    
    // do not write ghost cells
    if(i>1 and j>1 and i < imax-2 and j < jmax-2 and
       tx>0 and ty>0 and tx<BLOCK_DIMX-1 and ty<BLOCK_DIMY-1)
      {
	B[index+k*dz] = 
	  o2Gpu.R3 *   shmem1[zz2][ty+1][tx+1] + 
	  o2Gpu.R  * ( shmem1[zz2][ty+1][tx  ] +
		       shmem1[zz2][ty+1][tx+2] +
		       shmem1[zz2][ty  ][tx+1] +
		       shmem1[zz2][ty+2][tx+1] +
		       shmem1[zz1][ty+1][tx+1] +
		       shmem1[zz3][ty+1][tx+1] );
      }
    __syncthreads();

    // swap planes in shmem0
    tmp = z0;
    z0  = z1;
    z1  = z2;
    z2  = z3;
    z3  = z4;
    z4  = tmp;
    
    // swap planes in shmem1
    tmp  = zz1;
    zz1  = zz2;
    zz2  = zz3;
    zz3  = tmp;
    
    // update new top data in shmem0 at z4
    if( i<imax and j<jmax )
      {
	// bulk data
	shmem0[z4][ty+1][tx+1] = A[index+(k+3)*dz];

	// fill shared memory ghost cells (left and right)
	if (tx == 0) 
	  {
	    shmem0[z4][ty+1][0] = A[index-1+(k+3)*dz];// x-1
	    
	    if (i+BLOCK_DIMX < imax) { // x+BLOCK_DIMX 
	      shmem0[z4][ty+1][BLOCK_DIMX+1] = A[index+BLOCK_DIMX+(k+3)*dz];
	    }
	  }
	
	// fill shared memory ghost cells (up and down)
	if (ty == 0)
	  {
	    shmem0[z4][0][tx+1] = A[index-dy+(k+3)*dz];// y-1
	    if (j+BLOCK_DIMY < jmax) {
	      shmem0[z4][BLOCK_DIMY+1][tx+1] = A[index+BLOCK_DIMY*dy+(k+3)*dz];  // y+BLOCK_DIMY
	    }
	  }

      }
    __syncthreads();

    // compute first time step in shmem1 at z3
    if( i>0      and j>0      and 
	i<imax-1 and j<jmax-1) {

      shmem1[zz3][ty+1][tx+1] = // z3 plane
	o2Gpu.R3 *  shmem0[z3][ty+1][tx+1] +
	o2Gpu.R  * (shmem0[z3][ty+1][tx  ] +
		    shmem0[z3][ty+1][tx+2] +
		    shmem0[z3][ty  ][tx+1] +
		    shmem0[z3][ty+2][tx+1] +
		    shmem0[z2][ty+1][tx+1] +
		    shmem0[z4][ty+1][tx+1] 
		    );

    }

    // now we are ready to move one z-plane forward !!

  } // end for k

} // heat3d_ftcs_sharedmem4_order2_kernel


#endif // _HEAT3D_KERNEL_GPU_SHMEM4_H_
