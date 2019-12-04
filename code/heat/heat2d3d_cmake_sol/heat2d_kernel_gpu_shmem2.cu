/**
 * \file heat2d_kernel_gpu_shmem2.cu
 * \brief Implementation of CUDA kernel to solve 2D heat equation.
 *
 * \author Pierre Kestener
 * \date 18-dec-2009
 */

#ifndef _HEAT2D_KERNEL_GPU_SHMEM2_H_
#define _HEAT2D_KERNEL_GPU_SHMEM2_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX 16
#define BLOCK_DIMY 16

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
// What happens if you swap tx and ty in terms of performance
__global__ void heat2d_ftcs_sharedmem2_order2_kernel(real_t* A, real_t* B, 
						     int isize, int jsize)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index (avoid external ghost cells)
  const int i = __mul24(bx, blockDim.x) + tx + 1;
  const int j = __mul24(by, blockDim.y) + ty + 1;
  
  const int index = isize*j + i;
  
  // copy data from global mem to shared mem 
  // 2 ghost cells per dimension (1 left + 1 right)
  __shared__ real_t shmem[BLOCK_DIMY+2][BLOCK_DIMX+2];

  if( i<isize and j<jsize )
    {
      
      // fill bulk data
      shmem[ty+1][tx+1] = A[index];
    
      // fill shared memory ghost cells (left and right)
      if (tx == 0) 
	{
	  shmem[ty+1][tx] = A[index-1];          // x-1
	  if (i+BLOCK_DIMX < isize)
	    shmem[ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX]; // x+BLOCK_DIMX 
	}
      
      // fill shared memory ghost cells (up and down)
      if (ty == 0)
	{
	  shmem[ty][tx+1] = A[index-isize];             // y-1
	  if (j+BLOCK_DIMY < jsize)
	    shmem[ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*isize];  // y+BLOCK_DIMY
	}
    }
  // wait for all threads in the block to finish loading data
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if(i < isize-1 and j < jsize-1)
    {
      B[index]= o2Gpu.R2*shmem[ty+1][tx+1] + 
  	o2Gpu.R*(shmem[ty+1][tx  ]+shmem[ty+1][tx+2]+
		 shmem[ty  ][tx+1]+shmem[ty+2][tx+1]);
    }
  
} // heat2d_ftcs_sharedmem2_order2_kernel

#define BLOCK_DIMX2 64
#define BLOCK_DIMY2 6

/**
 * CUDA kernel for 4th order 2D heat solver using GPU shared memory 
 */
__global__ void heat2d_ftcs_sharedmem2_order4_kernel(real_t* A, real_t* B, 
						     int isize, int jsize)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = __mul24(bx, blockDim.x) + tx + 2;
  const int j = __mul24(by, blockDim.y) + ty + 2;
  
  const int index = isize*j + i;
  
  // copy data from global mem to shared mem 
  // 4 ghost cells per dimension (2 left + 2 right)
  __shared__ real_t shmem[BLOCK_DIMY2+4][BLOCK_DIMX2+4];

  if( i<isize and j<jsize )
    {
      // fill bulk data
      shmem[ty+2][tx+2] = A[index];
      
      // fill shared memory ghost cells along X
      if (tx == 0 or tx == 1) 
      	{
      	  shmem[ty+2][tx] = A[index-2];          // x-2
      	  if (i+BLOCK_DIMX2 < isize)
      	    shmem[ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2]; // x+BLOCK_DIMX2 
      	}
     
      // fill shared memory ghost cells along Y
      if (ty == 0 or ty == 1)
      	{
      	  shmem[ty][tx+2] = A[index-2*isize];             // y-2
      	  if (j+BLOCK_DIMY2 < jsize)
      	    shmem[ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize];  // y+BLOCK_DIMY2
      	}
    }    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  // notice that constraint i>=2 and j>= 2 are already met
  if( i < isize-2 and j < jsize-2 )
    {
      B[index]= o4Gpu.S2*shmem[ty+2][tx+2] + 
	o4Gpu.S*(-shmem[ty  ][tx+2]+16*shmem[ty+1][tx+2]+16*shmem[ty+3][tx+2]-shmem[ty+4][tx+2]
		 -shmem[ty+2][tx  ]+16*shmem[ty+2][tx+1]+16*shmem[ty+2][tx+3]-shmem[ty+2][tx+4]);
    }
  
} // heat2d_ftcs_sharedmem_order4_kernel

#endif // _HEAT2D_KERNEL_GPU_SHMEM2_H_
