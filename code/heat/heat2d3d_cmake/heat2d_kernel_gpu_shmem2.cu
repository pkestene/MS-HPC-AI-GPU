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

#define BLOCK_DIMX 64
#define BLOCK_DIMY 6

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
// que se passe-t-il en terme de performance et profiling si on echange tx et ty ?
__global__ void heat2d_ftcs_sharedmem2_order2_kernel(real_t* A, real_t* B, 
						     int pitch, int imax, int jmax)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index (avoid external ghost cells)
  const int i = /* TODO */;
  const int j = /* TODO */
  
    const int index = /* TODO */;
  
  // copy data from global mem to shared mem 
  // 2 ghost cells per dimension (1 left + 1 right)
  __shared__ real_t shmem[/* TODO */][/* TODO */];

  if( /* TODO */ )
    {
      
      // fill bulk data
      shmem[tx+1][ty+1] = A[index];
    
      // fill shared memory ghost cells (left and right)
      if (tx == 0) 
	{
	  /* TOOD */;
	}
      
      // fill shared memory ghost cells (up and down)
      if (ty == 0)
	{
	  /* TODO */;
	}
    }
  // wait for all threads in the block to finish loading data
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if( /* TODO */ )
    {
      B[index]= o2Gpu.R2*shmem[tx+1][ty+1] + 
  	o2Gpu.R*(shmem[tx  ][ty+1]+shmem[tx+2][ty+1]+
		 shmem[tx+1][ty  ]+shmem[tx+1][ty+2]);
    }
  
} // heat2d_ftcs_sharedmem2_order2_kernel

#define BLOCK_DIMX2 64
#define BLOCK_DIMY2 6

/**
 * CUDA kernel for 4th order 2D heat solver using GPU shared memory 
 */
__global__ void heat2d_ftcs_sharedmem2_order4_kernel(real_t* A, real_t* B, 
						     int pitch, int imax, int jmax)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // global index
  const int i = /* TODO */;
  const int j = /* TODO */;
  
  const int index = pitch*j + i;
  
  // copy data from global mem to shared mem 
  // 4 ghost cells per dimension (2 left + 2 right)
  __shared__ real_t shmem[/* TODO */][/* TODO */];

  if( /* TODO */ )
    {
      // fill bulk data
      shmem[/* TODO */][/* TODO */] = A[index];
      
      // fill shared memory ghost cells along X
      if (tx == 0 or tx == 1) 
      	{
	  /* TODO */;
      	}
     
      // fill shared memory ghost cells along Y
      if (ty == 0 or ty == 1)
      	{
	  /* TODO */;
      	}
    }    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  // notice that constraint i>=2 and j>= 2 are already met
  if( /* TODO */ )
    {
      B[index]= o4Gpu.S2*shmem[tx+2][ty+2] + 
	o4Gpu.S*(-shmem[tx+2][ty  ]+16*shmem[tx+2][ty+1]+16*shmem[tx+2][ty+3]-shmem[tx+2][ty+4]
		 -shmem[tx  ][ty+2]+16*shmem[tx+1][ty+2]+16*shmem[tx+3][ty+2]-shmem[tx+4][ty+2]);
    }

} // heat2d_ftcs_sharedmem_order4_kernel

#endif // _HEAT2D_KERNEL_GPU_SHMEM2_H_
