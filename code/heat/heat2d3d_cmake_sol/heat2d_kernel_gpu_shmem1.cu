/**
 * \file heat2d_kernel_gpu_shmem1.cu
 * \brief Implementation of CUDA kernel to solve 2D heat equation.
 *
 * \author Pierre Kestener
 * \date 18-dec-2009
 */

#ifndef _HEAT2D_KERNEL_GPU_SHMEM1_H_
#define _HEAT2D_KERNEL_GPU_SHMEM1_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX		64
#define BLOCK_DIMY		10
#define BLOCK_INNER_DIMX	(BLOCK_DIMX-2)
#define BLOCK_INNER_DIMY	(BLOCK_DIMY-2)


/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
// What happens if you swap tx and ty in terms of performance
__global__ void heat2d_ftcs_sharedmem_order2_kernel(real_t* A, real_t* B, 
						    int isize, int jsize)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, BLOCK_INNER_DIMY) + ty;
  
  const int index = isize*j + i;
  
  // copy data from global mem to shared mem (take care of bank conflicts)
  __shared__ real_t shmem[BLOCK_DIMY][BLOCK_DIMX];
  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      shmem[ty][tx] = A[index];
    }
    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if(tx>0 and tx<BLOCK_DIMX-1 and 
     ty>0 and ty<BLOCK_DIMY-1 and 
     i>0  and i < isize-1 and 
     j>0  and j < jsize-1)
    {
      B[index]= o2Gpu.R2*shmem[ty][tx] + 
	o2Gpu.R*(shmem[ty-1][tx]+shmem[ty+1][tx])+
	o2Gpu.R*(shmem[ty][tx-1]+shmem[ty][tx+1]);
    }

} // heat2d_ftcs_sharedmem_order2_kernel

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory
 * and using a different stencil.
 */
__global__ void heat2d_ftcs_sharedmem_order2b_kernel(real_t* A, real_t* B, 
						     int isize, int jsize)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, BLOCK_INNER_DIMX) + tx;
  const int j = __mul24(by, BLOCK_INNER_DIMY) + ty;
  
  const int index = isize*j + i;
  
  // copy data from global mem to shared mem (take care of bank conflicts)
  __shared__ real_t shmem[BLOCK_DIMY][BLOCK_DIMX+1];

  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      shmem[ty][tx] = A[index];
    }
    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if(tx>0 and tx<BLOCK_DIMX-1 and 
     ty>0 and ty<BLOCK_DIMY-1 and 
     i>0  and i < isize-1 and 
     j>0  and j < jsize-1)
    {
      real_t b = o2Gpu.R2b*shmem[ty][tx];

      b += o2Gpu.R*(shmem[ty-1][tx-1]);
      b += o2Gpu.R*(shmem[ty-1][tx  ]);
      b += o2Gpu.R*(shmem[ty-1][tx+1]);

      b += o2Gpu.R*(shmem[ty  ][tx-1]);
      b += o2Gpu.R*(shmem[ty  ][tx  ]);
      b += o2Gpu.R*(shmem[ty  ][tx+1]);

      b += o2Gpu.R*(shmem[ty+1][tx-1]);
      b += o2Gpu.R*(shmem[ty+1][tx  ]);
      b += o2Gpu.R*(shmem[ty+1][tx+1]);

      B[index]=b;
    }

} // heat2d_ftcs_sharedmem_order2b_kernel

#define BLOCK_DIMX2		20
#define BLOCK_DIMY2		18
#define BLOCK_INNER_DIMX2	(BLOCK_DIMX2-4)
#define BLOCK_INNER_DIMY2	(BLOCK_DIMY2-4)


/**
 * CUDA kernel for 4th order 2D heat solver using GPU shared memory 
 */
__global__ void heat2d_ftcs_sharedmem_order4_kernel(real_t* A, real_t* B, 
						    int isize, int jsize)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, BLOCK_INNER_DIMX2) + tx;
  const int j = __mul24(by, BLOCK_INNER_DIMY2) + ty;
  
  const int index = isize*j + i;
  
  // copy data from global mem to shared mem (take care of bank conflicts)
  __shared__ real_t shmem[BLOCK_DIMY2][BLOCK_DIMX2+1];
  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      shmem[ty][tx] = A[index];
    }
    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if(tx>1 and tx<BLOCK_DIMX2-2 and 
     ty>1 and ty<BLOCK_DIMY2-2 and
      i>1 and  i<isize-2        and
      j>1 and  j<jsize-2)
    {
      B[index]= o4Gpu.S2*shmem[ty][tx] + 
	o4Gpu.S*(-shmem[ty-2][tx]+16*shmem[ty-1][tx]+16*shmem[ty+1][tx]-shmem[ty+2][tx])+
	o4Gpu.S*(-shmem[ty][tx-2]+16*shmem[ty][tx-1]+16*shmem[ty][tx+1]-shmem[ty][tx+2]);
    }

} // heat2d_ftcs_sharedmem_order4_kernel

#endif // _HEAT2D_KERNEL_GPU_SHMEM1_H_
