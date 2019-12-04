/**
 * \file heat3d_kernel_gpu_shmem2.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * \author Pierre Kestener
 * \date 18-dec-2009
 */

#ifndef _HEAT3D_KERNEL_GPU_SHMEM2_H_
#define _HEAT3D_KERNEL_GPU_SHMEM2_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX 32
#define BLOCK_DIMY 16

/**
 * CUDA kernel for 2nd order 3D heat solver using GPU shared memory 
 */
__global__ void heat3d_ftcs_sharedmem2_order2_kernel(const real_t* __restrict__ A, 
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
  
  // global index (avoid external ghost cells)
  const int i = __mul24(bx, blockDim.x) + tx + 1;
  const int j = __mul24(by, blockDim.y) + ty + 1;
  
  const int index = isize*j + i;
  const int ijsize= isize*jsize;

  int low, mid, top, tmp;
  low=0;
  mid=1;
  top=2;
  
  // copy data from global mem to shared mem 
  // 2 ghost cells per dimension (1 left + 1 right)
  __shared__ real_t shmem[3][BLOCK_DIMY+2][BLOCK_DIMX+2];

  if( i<isize and j<jsize )
    {
      
      // fill bulk data
      shmem[low][ty+1][tx+1] = A[index+0*ijsize];
      shmem[mid][ty+1][tx+1] = A[index+1*ijsize];
      shmem[top][ty+1][tx+1] = A[index+2*ijsize];

  
      // fill shared memory ghost cells (left and right)
      if (tx == 0) 
	{
	  shmem[low][ty+1][tx] = A[index-1+0*ijsize];// x-1
	  shmem[mid][ty+1][tx] = A[index-1+1*ijsize];// x-1
	  shmem[top][ty+1][tx] = A[index-1+2*ijsize];// x-1

	  if (i+BLOCK_DIMX < isize) { // x+BLOCK_DIMX
	    shmem[low][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+0*ijsize];
	    shmem[mid][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+1*ijsize];
	    shmem[top][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+2*ijsize];
	  }
	}
      
      // fill shared memory ghost cells (up and down)
      if (ty == 0)
	{
	  shmem[low][ty][tx+1] = A[index-isize+0*ijsize];// y-1
	  shmem[mid][ty][tx+1] = A[index-isize+1*ijsize];// y-1
	  shmem[top][ty][tx+1] = A[index-isize+2*ijsize];// y-1
	  if (j+BLOCK_DIMY < jsize) {  // y+BLOCK_DIMY
	    shmem[low][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*isize+0*ijsize];
	    shmem[mid][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*isize+1*ijsize];
	    shmem[top][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*isize+2*ijsize];
	  }
	}
    }
  // wait for all threads in the block to finish loading data
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=1; k<ksize-1; ++k) {
    
    if(i < isize-1 and j < jsize-1)
      {
	B[index+k*ijsize] = 
	  o2Gpu.R3*shmem[mid][ty+1][tx+1] + 
	  o2Gpu.R*(shmem[mid][ty+1][tx  ] +
		   shmem[mid][ty+1][tx+2] +
		   shmem[mid][ty  ][tx+1] +
		   shmem[mid][ty+2][tx+1] +
		   shmem[low][ty+1][tx+1] +
		   shmem[top][ty+1][tx+1] );
      }
    __syncthreads();

    // swap planes
    tmp = low;
    low = mid;
    mid = top;
    top = tmp;
    
    // update new top data
    if( (j<jsize) and (i<isize) and (k<ksize-2) )
      {
	
	shmem[top][ty+1][tx+1] = A[index+(k+2)*ijsize];
	  
	// fill shared memory ghost cells (left and right)
	if (tx == 0) 
	  {
	    shmem[top][ty+1][tx] = A[index-1+(k+2)*ijsize];// x-1
	    
	    if (i+BLOCK_DIMX < isize) {
	      shmem[top][ty+1][tx+1+BLOCK_DIMX] = A[index+BLOCK_DIMX+(k+2)*ijsize]; // x+BLOCK_DIMX 
	    }
	  }
	
	// fill shared memory ghost cells (up and down)
	if (ty == 0)
	  {
	    shmem[top][ty][tx+1] = A[index-isize+(k+2)*ijsize];// y-1
	    if (j+BLOCK_DIMY < jsize) {
	      shmem[top][ty+1+BLOCK_DIMY][tx+1] = A[index+BLOCK_DIMY*isize+(k+2)*ijsize];  // y+BLOCK_DIMY
	    }
	  }

      } // end update new top data
    
    __syncthreads();

  } // end for k

} // heat3d_ftcs_sharedmem2_order2_kernel

#define BLOCK_DIMX2 32
#define BLOCK_DIMY2 16

/**
 * CUDA kernel for 4th order 3D heat solver using GPU shared memory 
 */
__global__ void heat3d_ftcs_sharedmem2_order4_kernel(const real_t* __restrict__ A, 
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
  
  // global index (avoid external ghost cells)
  const int i = __mul24(bx, blockDim.x) + tx + 2;
  const int j = __mul24(by, blockDim.y) + ty + 2;
  
  const int index = isize*j + i;
  const int ijsize= isize*jsize;
  
  // z-plane indexes
  int low, mid1, mid2, mid3, top, tmp;
  low =0;
  mid1=1;
  mid2=2;
  mid3=3;
  top =4;

  // copy data from global mem to shared mem 
  // 4 ghost cells per dimension (2 left + 2 right)
  // will hold data in 5 planes (z=k-2,k-1,k,k+1,k+2)
  __shared__ real_t shmem[5][BLOCK_DIMY2+4][BLOCK_DIMX2+4];

  // load all we need to start computation
  if( i<isize and j<jsize )
    {

      shmem[low ][ty+2][tx+2] = A[index+0*ijsize]; // data@ z=0
      shmem[mid1][ty+2][tx+2] = A[index+1*ijsize]; // data@ z=1
      shmem[mid2][ty+2][tx+2] = A[index+2*ijsize]; // data@ z=2
      shmem[mid3][ty+2][tx+2] = A[index+3*ijsize]; // data@ z=3
      shmem[top ][ty+2][tx+2] = A[index+4*ijsize]; // data@ z=4
      
      // fill shared memory ghost cells along X in mid plane
      if (tx == 0 or tx == 1) 
      	{
      	  shmem[low] [ty+2][tx] = A[index-2+0*ijsize];    // x-2
      	  shmem[mid1][ty+2][tx] = A[index-2+1*ijsize];    // x-2
      	  shmem[mid2][ty+2][tx] = A[index-2+2*ijsize];    // x-2
      	  shmem[mid3][ty+2][tx] = A[index-2+3*ijsize];    // x-2
      	  shmem[top ][ty+2][tx] = A[index-2+4*ijsize];    // x-2
      	  if (i+BLOCK_DIMX2 < isize) { // x+BLOCK_DIMX2
      	    shmem[low ][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+0*ijsize];
      	    shmem[mid1][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+1*ijsize];
      	    shmem[mid2][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+2*ijsize];
      	    shmem[mid3][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+3*ijsize];
      	    shmem[top ][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+4*ijsize];
	  }
      	}
     
      // fill shared memory ghost cells along Y in mid plane
      if (ty == 0 or ty == 1)
      	{
      	  shmem[low ][ty][tx+2] = A[index-2*isize+0*ijsize];  // y-2
      	  shmem[mid1][ty][tx+2] = A[index-2*isize+1*ijsize];  // y-2
      	  shmem[mid2][ty][tx+2] = A[index-2*isize+2*ijsize];  // y-2
      	  shmem[mid3][ty][tx+2] = A[index-2*isize+3*ijsize];  // y-2
      	  shmem[top ][ty][tx+2] = A[index-2*isize+4*ijsize];  // y-2
      	  if (j+BLOCK_DIMY2 < jsize) { // y+BLOCK_DIMY2
      	    shmem[low ][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+0*ijsize];
      	    shmem[mid1][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+1*ijsize];
      	    shmem[mid2][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+2*ijsize];
      	    shmem[mid3][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+3*ijsize];
      	    shmem[top ][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+4*ijsize];
	  }
      	}
    } // end load initial data
  __syncthreads();
  
  

  // do FTCS time step update and copy back results to global mem buffer B
  // notice that constraint i>=2 and j>= 2 are already met
  for (int k=2; k<ksize-2; ++k) {
    
    if( i < isize-2 and j < jsize-2 )
      {
	real_t value;
	 
	value =  o4Gpu.S3 *    shmem[mid2][ty+2][tx+2];

	value += o4Gpu.S  *(-  shmem[mid2][ty  ][tx+2]
			  + 16*shmem[mid2][ty+1][tx+2]
			  + 16*shmem[mid2][ty+3][tx+2]
			  -    shmem[mid2][ty+4][tx+2]);

	value += o4Gpu.S  *(-  shmem[mid2][ty+2][tx  ]
			  + 16*shmem[mid2][ty+2][tx+1]
			  + 16*shmem[mid2][ty+2][tx+3]
			  -    shmem[mid2][ty+2][tx+4]);

	value += o4Gpu.S  *(-  shmem[low ][ty+2][tx+2]
			  + 16*shmem[mid1][ty+2][tx+2]
			  + 16*shmem[mid3][ty+2][tx+2]
			  -    shmem[top ][ty+2][tx+2]);
	
	// finally, write result back to global memory
	B[index+k*ijsize] = value;
      }
    __syncthreads();

    // rotate shared memory planes
    tmp  = low;
    low  = mid1;
    mid1 = mid2;
    mid2 = mid3;
    mid3 = top;
    top  = tmp;
    
    // update new top data
    if( (j<jsize) and (i<isize) and (k<ksize-3) )
      {
	shmem[top ][ty+2][tx+2] = A[index+(k+3)*ijsize];

	// fill shared memory ghost cells (left and right along x)
	if (tx == 0 or tx == 1) 
	  {
	    shmem[top][ty+2][tx] = A[index-2+(k+3)*ijsize];// x-2
	    
	    if (i+BLOCK_DIMX2 < isize) {
	      shmem[top][ty+2][tx+2+BLOCK_DIMX2] = A[index+BLOCK_DIMX2+(k+3)*ijsize]; // x+BLOCK_DIMX2 
	    }
	  } // end tx == 0 or tx == 1

	// fill shared memory ghost cells (left and right along y)
	if (ty == 0 or ty == 1) 
	  {
	    shmem[top][ty][tx+2] = A[index-2*isize+(k+3)*ijsize];// y-2
	    
	    if (j+BLOCK_DIMY2 < jsize) {
	      shmem[top][ty+2+BLOCK_DIMY2][tx+2] = A[index+BLOCK_DIMY2*isize+(k+3)*ijsize]; // y+BLOCK_DIMY2 
	    }
	  } // end ty == 0 or ty == 1

      } // end update new top data

    __syncthreads();

  } // end for k

} // heat3d_ftcs_sharedmem_order4_kernel

#endif // _HEAT3D_KERNEL_GPU_SHMEM2_H_
