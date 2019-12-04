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
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index,index1,index2,index3,index4;
  index = j*NX + i;
  index1= index + 1;
  index2= index - 1;
  index3= index + NX;
  index4= index - NX;


  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
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
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index;
  int indexX1,indexX2,indexX3,indexX4;
  int indexY1,indexY2,indexY3,indexY4;
  index = j*NX + i;
  indexX1= index - 2;
  indexX2= index - 1;
  indexX3= index + 1;
  indexX4= index + 2;
  
  indexY1= index - 2*NX;
  indexY2= index - NX;
  indexY3= index + NX;
  indexY4= index + 2*NX;

  if(i>1 && i<NX-2 && j>1 && j<NY-2) {
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
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index;
  index = j*NX + i;

  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    real_t b = R2b*A[index]; 
    
    b += R * A[index+1-NX];
    b += R * A[index  -NX];
    b += R * A[index-1-NX];

    b += R * A[index+1];
    b += R * A[index  ];
    b += R * A[index-1];
   
    b += R * A[index+1+NX];
    b += R * A[index  +NX];
    b += R * A[index-1+NX];
   
   
    B[index] = b;
  }
  
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
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index,index1,index2,index3,index4;
  index = j*NX + i;
  index1= index + 1;
  index2= index - 1;
  index3= index + NX;
  index4= index - NX;

  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    real_t tmp = R2*A[index] + R*( A[index1] + A[index2] + A[index3] + A[index4] );
    B[index] = tmp*mask[index]+1-mask[index];
  }
  
} // heat2d_ftcs_naive_order2_mask_kernel


#endif // _HEAT2D_KERNEL_GPU_NAIVE_H_
