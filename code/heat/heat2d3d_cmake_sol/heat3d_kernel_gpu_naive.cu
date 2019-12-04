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
__global__ void heat3d_ftcs_naive_order2_kernel(const real_t *__restrict__ A, 
						real_t *B, 
						unsigned int NX,
						unsigned int NY,
						unsigned int NZ,
						real_t R,
						real_t R3)
{
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index;

  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
  
    for (int k=1; k<NZ-1; ++k) {
      
      index = (k*NY+j)*NX + i;
      real_t tmp  = R3*A[index] +
	R * ( A[index+1]     + A[index-1]  +
	      A[index+NX]    + A[index-NX] +
	      A[index+NX*NY] + A[index-NX*NY]);
      
      B[index] = tmp;

    }
  }
  
} // heat3d_ftcs_naive_order2_kernel

// naive kernel, everything in global memory (FCTS order 2b)
__global__ void heat3d_ftcs_naive_order2b_kernel(const real_t *__restrict__ A, 
						 real_t *B, 
						 unsigned int NX,
						 unsigned int NY,
						 unsigned int NZ,
						 real_t R,
						 real_t R3)
{
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index;
  int offset = i+NX*j;

  if(i>0 && i<NX-1 && j>0 && j<NY-1) {
    
    for (int k=1; k<NZ-1; ++k) {
      
      index = offset + k*NX*NY;
      real_t tmp = R3*A[index];

      // 3x3 stencil
      for (int kk=-1; kk<2; kk++)
	for (int jj=-1; jj<2; jj++)
	  for (int ii=-1; ii<2; ii++) {
	    int index2 = index + ii + NX*jj + NX*NY*kk;
	      tmp += R*A[index2];
	    }
      
      B[index] = tmp;

    }
  }
  
} // heat3d_ftcs_naive_order2b_kernel

// naive kernel, everything in global memory (FCTS order 4)
__global__ void heat3d_ftcs_naive_order4_kernel(const real_t *__restrict__ A, 
						real_t *B, 
						unsigned int NX,
						unsigned int NY,
						unsigned int NZ,
						real_t S,
						real_t S3)
{
  
  int i = blockIdx.x * blockDim.x + threadIdx.x ;
  int j = blockIdx.y * blockDim.y + threadIdx.y ;
  
  // column-major order
  int index;
  //int indexX1,indexX2,indexX3,indexX4;
  //int indexY1,indexY2,indexY3,indexY4;
  //int indexZ1,indexZ2,indexZ3,indexZ4;
 
  const int NXY = NX*NY;

  if(i>1 and i<NX-2 and 
     j>1 and j<NY-2) {

    for (int k=2; k<NZ-2; ++k) {
      
      index = (k*NY+j)*NX + i;
      real_t tmp = S3*A[index];

      tmp += (-1)*S*A[index-2];
      tmp += (16)*S*A[index-1];
      tmp += (16)*S*A[index+1];
      tmp += (-1)*S*A[index+2];

      tmp += (-1)*S*A[index-2*NX];
      tmp += (16)*S*A[index-1*NX];
      tmp += (16)*S*A[index+1*NX];
      tmp += (-1)*S*A[index+2*NX];

      tmp += (-1)*S*A[index-2*NXY];
      tmp += (16)*S*A[index-1*NXY];
      tmp += (16)*S*A[index+1*NXY];
      tmp += (-1)*S*A[index+2*NXY];

      B[index] = tmp; 

    }
  }
  
} // heat3d_ftcs_naive_order4_kernel

#endif // _HEAT3D_KERNEL_GPU_NAIVE_H_
