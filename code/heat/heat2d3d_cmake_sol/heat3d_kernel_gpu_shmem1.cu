/**
 * \file heat3d_kernel_gpu_shmem1.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * \date 27-dec-2009
 */

#ifndef _HEAT3D_KERNEL_GPU_SHMEM1_H_
#define _HEAT3D_KERNEL_GPU_SHMEM1_H_

#include <stdio.h> // for printf

#define BLOCK_DIMX		32
#define BLOCK_DIMY		16
#define BLOCK_INNER_DIMX	(BLOCK_DIMX-2)
#define BLOCK_INNER_DIMY	(BLOCK_DIMY-2)

#include "param.h"

// What happens if you swap tx and ty in terms of performance ?
__global__ void heat3d_ftcs_sharedmem_order2_kernel(const real_t* __restrict__ A, 
						    real_t* __restrict__ B, 
						    int isize, 
						    int jsize, 
						    int ksize)
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
  const int ijsize = isize*jsize;

  int low, mid, top, tmp;
  low=0;
  mid=1;
  top=2;

  // load the first 3 planes from global mem to shared mem
  __shared__ real_t shmem[3][BLOCK_DIMY][BLOCK_DIMX/*+1*/];

  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      shmem[low][ty][tx] = A[index];
      shmem[mid][ty][tx] = A[index+  ijsize];
      shmem[top][ty][tx] = A[index+2*ijsize];
    }
    
  __syncthreads();
  

  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=1; k<ksize-2; ++k) {

    if(tx>0 and tx<BLOCK_DIMX-1 and 
       ty>0 and ty<BLOCK_DIMY-1 and 
       i>0  and i < isize-1 and 
       j>0  and j < jsize-1)
      {

	B[index+k*ijsize]= 
	  o2Gpu.R3*shmem[mid][ty][tx] + 
	  o2Gpu.R*(shmem[mid][ty-1][tx]+shmem[mid][ty+1][tx]+
		   shmem[mid][ty][tx-1]+shmem[mid][ty][tx+1]+
		   shmem[top][ty][tx]  +shmem[low][ty][tx]);
	
      }
    
    __syncthreads();
    
    // swap planes
    tmp = low;
    low = mid;
    mid = top;
    top = tmp;
    
    // update new top data
    if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
      {
	shmem[top][ty][tx] = A[index+(k+2)*ijsize];
      }
    
    __syncthreads();
  } // end for (k ...
  
  // last plane
  int k = ksize-2;
  if(tx>0 and tx<BLOCK_DIMX-1 and 
     ty>0 and ty<BLOCK_DIMY-1 and 
     i>0  and i < isize-1 and 
     j>0  and j < jsize-1)
    {
      B[index+k*ijsize]= 
	o2Gpu.R3*shmem[mid][ty  ][tx  ]+ 
	o2Gpu.R*(shmem[mid][ty-1][tx  ]+shmem[mid][ty+1][tx  ]+
		 shmem[mid][ty  ][tx-1]+shmem[mid][ty  ][tx+1]+
		 shmem[top][ty  ][tx  ]+shmem[low][ty  ][tx  ]);
    }

} // heat3d_ftcs_sharedmem_order2_kernel


/*
 * Same kernel but with data prefetch
 */
__global__ void heat3d_ftcs_sharedmem_order2_dataprefetch_kernel(const real_t* __restrict__ A, 
								 real_t* __restrict__ B, 
								 int isize, 
								 int jsize, 
								 int ksize)
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
  
  const int index  = isize*j + i;
  const int ijsize = isize*jsize;
  
  int low, mid, top;
  low=0;
  mid=1;
  top=2;

  real_t nextPlaneData;

  // load the first 3 planes from global mem to shared mem
  __shared__ real_t shmem[3][BLOCK_DIMY][BLOCK_DIMX+1];

  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      shmem[low][ty][tx] = A[index];
      shmem[mid][ty][tx] = A[index+  ijsize];
      nextPlaneData      = A[index+2*ijsize];
    }    
  __syncthreads();
  

  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=1; k<ksize-2; ++k) {

    // deposit next plane into shared mem
    if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
      {
	shmem[top][ty][tx] = nextPlaneData;
      }
    __syncthreads();

    // perform computation at z=k from data in shared memory
    if(tx>0 and tx<BLOCK_DIMX-1 and 
       ty>0 and ty<BLOCK_DIMY-1 and 
       i>0  and i < isize-1 and 
       j>0  and j < jsize-1)
      {

	B[index+k*ijsize] =
	  o2Gpu.R3*shmem[mid][ty][tx] + 
	  o2Gpu.R*(shmem[mid][ty-1][tx  ]+shmem[mid][ty+1][tx  ]+
		   shmem[mid][ty  ][tx-1]+shmem[mid][ty  ][tx+1]+
		   shmem[top][ty  ][tx  ]+shmem[low][ty  ][tx  ]);
	
      }
    __syncthreads();
    
    // rotate shared memory planes and load next plane data into register
    low = mid;
    mid = top;
    nextPlaneData =  A[index+(k+2)*ijsize];
    __syncthreads();

  } // end for (k ...
  __syncthreads();
  
  // last plane
  int k = ksize-2;
  if(tx>0 and tx<BLOCK_DIMX-1 and 
     ty>0 and ty<BLOCK_DIMY-1 and 
     i>0  and i < isize-1 and 
     j>0  and j < jsize-1)
    {
      B[index+k*ijsize]= 
	o2Gpu.R3*shmem[mid][ty  ][tx  ]+ 
	o2Gpu.R*(shmem[mid][ty-1][tx  ]+shmem[mid][ty+1][tx  ]+
		 shmem[mid][ty  ][tx-1]+shmem[mid][ty  ][tx+1]+
		 nextPlaneData         +shmem[low][ty  ][tx  ]);
    }

} // heat3d_ftcs_sharedmem_order2_kernel

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

#define BLOCK_DIMX2		32
#define BLOCK_DIMY2		16
#define BLOCK_INNER_DIMX2	(BLOCK_DIMX2-4)
#define BLOCK_INNER_DIMY2	(BLOCK_DIMY2-4)

#include "param.h"

/** 
 * CUDA Kernel for 4th order computation.
 * Balance between register and shared memory.
 */
__global__ void heat3d_ftcs_sharedmem_order4_kernel(const real_t* __restrict__ A, 
						    real_t* __restrict__ B, 
						    int isize, 
						    int jsize, 
						    int ksize)
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
  
  const int index  = isize*j + i;
  const int ijsize = isize*jsize;
  
  int low, mid, top;
  low=0;
  mid=1;
  top=2;

  real_t data_zm2;
  real_t data_zp2;

  // load the 3 planes (z=k-1,k,k+1) from global mem to shared mem
  __shared__ real_t shmem[3][BLOCK_DIMY2][BLOCK_DIMX2];

  // load all we need to start computation
  if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
    {
      data_zm2           = A[index];          // data@ z=0
      shmem[low][ty][tx] = A[index+  ijsize]; // data@ z=1
      shmem[mid][ty][tx] = A[index+2*ijsize]; // data@ z=2
      shmem[top][ty][tx] = A[index+3*ijsize]; // data@ z=3
      data_zp2           = A[index+4*ijsize]; // data@ z=4
    }
  __syncthreads();
  

  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=2; k<ksize-3; k++) {

    if(tx>1 and tx<BLOCK_DIMX2-2 and 
       ty>1 and ty<BLOCK_DIMY2-2 and 
       i >1 and i < isize-2      and 
       j >1 and j < jsize-2)
      {
	real_t tmp;

	tmp =  o4Gpu.S3*   shmem[mid][ty  ][tx  ];

	tmp += o4Gpu.S*(  -shmem[mid][ty-2][tx  ]+
			16*shmem[mid][ty-1][tx  ]+
			16*shmem[mid][ty+1][tx  ]
			  -shmem[mid][ty+2][tx  ]);
	tmp += o4Gpu.S*(  -shmem[mid][ty  ][tx-2]+
			16*shmem[mid][ty  ][tx-1]+
			16*shmem[mid][ty  ][tx+1]
			  -shmem[mid][ty  ][tx+2]);

	tmp += o4Gpu.S*(-  data_zm2 + 
			16*shmem[low][ty  ][tx  ] + 
			16*shmem[top][ty  ][tx  ]
			-  data_zp2);

	// finally, write result back to global memory
	B[index+k*ijsize]= tmp;
      }
    __syncthreads();
    
    // swap planes
    data_zm2           = shmem[low][ty][tx];
    shmem[low][ty][tx] = shmem[mid][ty][tx];
    shmem[mid][ty][tx] = shmem[top][ty][tx];
    shmem[top][ty][tx] = data_zp2;

    // update data_zp2 by loading data at z=k+3
    if((j>=0) and (j<jsize) and (i>=0) and (i<isize)) 
      {
	data_zp2 = A[index+(k+3)*ijsize];
      }
    
    __syncthreads();
  } // end for (k ...
  
  // last plane
  int k = ksize-3;
  if(tx>1 and tx<BLOCK_DIMX2-2 and 
     ty>1 and ty<BLOCK_DIMY2-2 and 
     i >1 and i < isize-2 and 
     j >1 and j < jsize-2)
    {
      real_t tmp = o4Gpu.S3*shmem[mid][ty][tx];
      tmp += o4Gpu.S*(  -shmem[mid][ty-2][tx  ]+
		      16*shmem[mid][ty-1][tx  ]+
		      16*shmem[mid][ty+1][tx  ]
			-shmem[mid][ty+2][tx  ]);
      tmp += o4Gpu.S*(  -shmem[mid][ty  ][tx-2]+
		      16*shmem[mid][ty  ][tx-1]+
		      16*shmem[mid][ty  ][tx+1]
			-shmem[mid][ty  ][tx+2]);
      tmp += o4Gpu.S*(-  data_zm2 + 
		      16*shmem[low][ty  ][tx  ] + 
		      16*shmem[top][ty  ][tx  ]
		      -  data_zp2);
      B[index+k*ijsize] = tmp;
    }

} // heat3d_ftcs_sharedmem_order4_kernel

#endif // _HEAT3D_KERNEL_GPU_SHMEM1_H_
