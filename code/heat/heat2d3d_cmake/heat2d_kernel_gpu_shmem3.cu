/**
 * \file heat2d_kernel_gpu_shmem3.cu
 * \brief Implementation of CUDA kernel to solve 2D heat equation.
 *
 * \date 18-dec-2009
 */

#ifndef _HEAT2D_KERNEL_GPU_SHMEM3_H_
#define _HEAT2D_KERNEL_GPU_SHMEM3_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

// Row kernel parameters
#define ROWS_BLOCK_X		16
#define ROWS_BLOCK_Y		4
#define ROWS_NBLOCKS            4

// Column kernel parameters
#define COLS_BLOCK_X	       16
#define COLS_BLOCK_Y	        8 // 4 ou 8
#define COLS_NBLOCKS            4

#include "param.h"

/** 
 * 
 * 
 * @param A      : input 2D buffer
 * @param B      : output 2D buffer
 * @param ipitch : pitch along i-dimension
 * @param imax   : i-size
 * @param jmax   : j-size
 */
__global__ void heat2d_ftcs_sharedmem3_row_kernel(real_t* A, real_t* B, 
						  int ipitch, int imax, int jmax)
{
  
 __shared__ real_t shmem[ROWS_BLOCK_Y][(ROWS_NBLOCKS + 2) * ROWS_BLOCK_X];
 
 // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // compute global index in left halo
  const int i = __mul24(bx, ROWS_NBLOCKS*ROWS_BLOCK_X) - ROWS_BLOCK_X + tx;
  const int j = __mul24(by, ROWS_BLOCK_Y)  + ty;
  
  A += ipitch*j + i;
  B += ipitch*j + i;

  // copy main data from global mem to shared mem
  #pragma unroll
  for(int k = 1; k <= ROWS_NBLOCKS; k++)
    shmem[ty][tx + k * ROWS_BLOCK_X] = A[k * ROWS_BLOCK_X];

  // copy left halo data
  shmem[ty][tx] = (i >= 0 ) ? A[0] : 0;
  
  // copy right halo data
  unsigned int k = ROWS_NBLOCKS+1;
  shmem[ty][tx+k*ROWS_BLOCK_X] = (imax - i > k * ROWS_BLOCK_X) ? A[k * ROWS_BLOCK_X] : 0;

  // synchonize threads inside block
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  #pragma unroll
  for(int k = 1; k < 1+ROWS_NBLOCKS; k++){
    real_t sum = 0.0;
    
    sum += o2Gpu.R2*shmem[ty][tx     + k * ROWS_BLOCK_X];
    sum += o2Gpu.R *shmem[ty][tx + 1 + k * ROWS_BLOCK_X];
    sum += o2Gpu.R *shmem[ty][tx - 1 + k * ROWS_BLOCK_X];

    if (j>0 and j<jmax-1 and
	i+k*ROWS_BLOCK_X>0 and i+k*ROWS_BLOCK_X<imax-1)
      B[k * ROWS_BLOCK_X] = sum;
  }
  
}

/** 
 * Swap raws and columns.
 * 
 * @param A      : input 2D buffer
 * @param B      : output 2D buffer
 * @param ipitch : pitch along i-dimension
 * @param imax   : i-size
 * @param jmax   : j-size
 */
__global__ void heat2d_ftcs_sharedmem3_col_kernel(real_t* A, real_t* B, 
						  int ipitch, int imax, int jmax)
{

  /* TODO */
  
}

#endif // _HEAT2D_KERNEL_GPU_SHMEM3_H_
