/**
 * \file heat3d_kernel_gpu_naive.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * \date 27-dec-2009
 */

#ifndef _HEAT3D_KERNEL_GPU_NAIVE_H_
#define _HEAT3D_KERNEL_GPU_NAIVE_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

// naive kernel, everything in global memory (FCTS order 2)
__global__ void heat3d_ftcs_naive_order2_kernel(real_t *A, real_t *B, 
						unsigned int NX,
						unsigned int NY,
						unsigned int NZ,
						real_t R,
						real_t R3)
{
  
  /* TODO */;
  
} // heat3d_ftcs_naive_order2_kernel

// naive kernel, everything in global memory (FCTS order 2b)
__global__ void heat3d_ftcs_naive_order2b_kernel(real_t *A, real_t *B, 
						 unsigned int NX,
						 unsigned int NY,
						 unsigned int NZ,
						 real_t R,
						 real_t R3)
{
  
  /* TODO */;
  
} // heat3d_ftcs_naive_order2b_kernel

// naive kernel, everything in global memory (FCTS order 4)
__global__ void heat3d_ftcs_naive_order4_kernel(real_t *A, real_t *B, 
						unsigned int NX,
						unsigned int NY,
						unsigned int NZ,
						real_t S,
						real_t S3)
{
  
  /* TODO */;
  
} // heat3d_ftcs_naive_order4_kernel

#endif // _HEAT3D_KERNEL_GPU_NAIVE_H_
