/**
 * \file heat3d_kernel_gpu_shmem3.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * Specific feature : perform 2 time steps in a single kernel
 *
 * \author Pierre Kestener
 * \date 20-nov-2012
 */

#ifndef _HEAT3D_KERNEL_GPU_SHMEM3_H_
#define _HEAT3D_KERNEL_GPU_SHMEM3_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX 24
#define BLOCK_DIMY 16
#define BLOCK_INNER_DIMX (BLOCK_DIMX-4)
#define BLOCK_INNER_DIMY (BLOCK_DIMY-4)

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
__global__ void heat3d_ftcs_sharedmem3_order2_kernel(real_t* A, real_t* B, 
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
  const int i = __mul24(bx, BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, BLOCK_INNER_DIMY) + ty;
  
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
  __shared__ real_t shmem0[5][BLOCK_DIMY][BLOCK_DIMX];
  // shmem1: temperature at t+1 : ghost cell not needed, but use them anyway
  // so that we can have the same indexes
  __shared__ real_t shmem1[3][BLOCK_DIMY][BLOCK_DIMX];

  /*
   * copy data from global mem to shared mem 
   */
  // initialize temperature at t (just copy from external memory)
  if( i<imax and j<jmax )
    {
      
      shmem0[z0][ty][tx] = A[index+z0*dz];
      shmem0[z1][ty][tx] = A[index+z1*dz];
      shmem0[z2][ty][tx] = A[index+z2*dz];
      shmem0[z3][ty][tx] = A[index+z3*dz];
      shmem0[z4][ty][tx] = A[index+z4*dz];

    }
  // wait for all threads in the block to finish loading data
  __syncthreads();

  // initialize temperature at time t+1
  if( tx>0      and tx<BLOCK_DIMX-1 and
      ty>0      and ty<BLOCK_DIMY-1 and 
      i<imax-1  and j<jmax-1 ) {

    shmem1[zz1][ty][tx] = // z1 plane
      o2Gpu.R3 *  shmem0[z1][ty  ][tx  ] +
      o2Gpu.R  * (shmem0[z1][ty  ][tx-1] +
		  shmem0[z1][ty  ][tx+1] +
		  shmem0[z1][ty-1][tx  ] +
		  shmem0[z1][ty+1][tx  ] +
		  shmem0[z0][ty  ][tx  ] +
		  shmem0[z2][ty  ][tx  ] 
		  );
    shmem1[zz2][ty][tx] = // z2 plane
      o2Gpu.R3 *  shmem0[z2][ty  ][tx  ] +
      o2Gpu.R  * (shmem0[z2][ty  ][tx-1] +
		  shmem0[z2][ty  ][tx+1] +
		  shmem0[z2][ty-1][tx  ] +
		  shmem0[z2][ty+1][tx  ] +
		  shmem0[z1][ty  ][tx  ] +
		  shmem0[z3][ty  ][tx  ] 
		  );
    shmem1[zz3][ty][tx] = // z3 plane
      o2Gpu.R3 *  shmem0[z3][ty  ][tx  ] +
      o2Gpu.R  * (shmem0[z3][ty  ][tx-1] +
		  shmem0[z3][ty  ][tx+1] +
		  shmem0[z3][ty-1][tx  ] +
		  shmem0[z3][ty+1][tx  ] +
		  shmem0[z2][ty  ][tx  ] +
		  shmem0[z4][ty  ][tx  ] 
		  );
  }
 
  // do FTCS time steps update and copy back results to global mem buffer B
  for (int k=2; k<kmax-3; ++k) {
    
    // do not write ghost cells
    if(i>1  and j>1  and i < imax-2      and j < jmax-2 and
       tx>1 and ty>1 and tx<BLOCK_DIMX-2 and ty<BLOCK_DIMY-2)
      {
	B[index+k*dz] = 
	  o2Gpu.R3 *   shmem1[zz2][ty  ][tx  ] + 
	  o2Gpu.R  * ( shmem1[zz2][ty  ][tx-1] +
		       shmem1[zz2][ty  ][tx+1] +
		       shmem1[zz2][ty-1][tx  ] +
		       shmem1[zz2][ty+1][tx  ] +
		       shmem1[zz1][ty  ][tx  ] +
		       shmem1[zz3][ty  ][tx  ] );
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

	shmem0[z4][ty][tx] = A[index+(k+3)*dz];

      }
    __syncthreads();

    // compute first time step in shmem1 at z3
    if( tx>0            and 
	tx<BLOCK_DIMX-1 and
	ty>0            and 
	ty<BLOCK_DIMY-1 and  
	i<imax-1        and 
	j<jmax-1) {

      shmem1[zz3][ty][tx] = // z3 plane
	o2Gpu.R3 *  shmem0[z3][ty  ][tx  ] +
	o2Gpu.R  * (shmem0[z3][ty  ][tx-1] +
		    shmem0[z3][ty  ][tx+1] +
		    shmem0[z3][ty-1][tx  ] +
		    shmem0[z3][ty+1][tx  ] +
		    shmem0[z2][ty  ][tx  ] +
		    shmem0[z4][ty  ][tx  ] 
		    );

    }

    // now we are ready to move one z-plane forward !!

  } // end for k

} // heat3d_ftcs_sharedmem3_order2_kernel

#endif // _HEAT3D_KERNEL_GPU_SHMEM3_H_
