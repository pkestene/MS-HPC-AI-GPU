/**
 * \file heat2d_kernel_gpu_naive.cu
 * \brief Implementation of CUDA kernel to solve 2D heat equation.
 *
 * \date 18-dec-2009
 */

#ifndef _HEAT2D_KERNEL_GPU_NAIVE_H_
#define _HEAT2D_KERNEL_GPU_NAIVE_H_

#include <stdio.h> // for printf
#include "param.h" // for real_t typedef

/**
 * naive kernel, everything in global memory
 */
__global__ void heat2d_ftcs_naive_order2_kernel(real_t *A, real_t *B, 
						unsigned int NX,
						unsigned int NY,
						real_t R,
						real_t R2)
{
  
  int i = /* TODO */;
  int j = /* TODO */;
  
  // column-major order
  int index,index1,index2,index3,index4;
  index = /* TODO */;
  index1= /* TODO */;
  index2= /* TODO */;
  index3= /* TODO */;
  index4= /* TODO */;

  if (/* TODO */) {
    B[index] = R2*A[index] + R*( A[index1] + A[index2] + A[index3] + A[index4] );
  }
  
} // heat2d_ftcs_naive_order2_kernel

/**
 * naive kernel, everything in global memory (FCTS order 4)
 */
__global__ void heat2d_ftcs_naive_order4_kernel(real_t *A, real_t *B, 
						unsigned int NX,
						unsigned int NY,
						real_t S,
						real_t S2)
{
  
  int i = /* TODO */;
  int j = /* TODO */;
  
  // column-major order
  int index;
  int indexX1,indexX2,indexX3,indexX4;
  int indexY1,indexY2,indexY3,indexY4;
  index = /* TODO */;
  indexX1= /* TODO */;
  indexX2= /* TODO */;
  indexX3= /* TODO */;
  indexX4= /* TODO */;
  
  indexY1= /* TODO */;
  indexY2= /* TODO */;
  indexY3= /* TODO */;
  indexY4= /* TODO */;

  if( /* TODO */ ) {
    B[index] = S2*A[index] + 
      S * ( -A[indexX1] + 16*A[indexX2] + 16*A[indexX3] - A[indexX4] ) +
      S * ( -A[indexY1] + 16*A[indexY2] + 16*A[indexY3] - A[indexY4] );
  }
  
} // heat2d_ftcs_naive_order4_kernel

/**
 * naive kernel, everything in global memory
 */
__global__ void heat2d_ftcs_naive_order2b_kernel(real_t *A, real_t *B, 
						 unsigned int NX,
						 unsigned int NY,
						 real_t R,
						 real_t R2b)
{

  /* TODO */;
  
} // heat2d_ftcs_naive_order2b_kernel

/**
 * naive kernel, everything in global memory, use mask (for GLUT gui)
 */
__global__ void heat2d_ftcs_naive_order2_mask_kernel(real_t *A, real_t *B,
						     int *mask,
						     unsigned int NX,
						     unsigned int NY,
						     real_t R,
						     real_t R2)
{
  
  /* TODO */;
  
} // heat2d_ftcs_naive_order2_mask_kernel


#endif // _HEAT2D_KERNEL_GPU_NAIVE_H_
