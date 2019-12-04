/**
 * \file heat3d_kernel_gpu_shmem1.cu
 * \brief Implementation of CUDA kernel to solve 3D heat equation.
 *
 * \date 27-dec-2009
 */

#ifndef _HEAT3D_KERNEL_GPU_SHMEM1_H_
#define _HEAT3D_KERNEL_GPU_SHMEM1_H_

#include <stdio.h> // for printf

#define BLOCK_DIMX		16
#define BLOCK_DIMY		30
#define BLOCK_INNER_DIMX	(BLOCK_DIMX-2)
#define BLOCK_INNER_DIMY	(BLOCK_DIMY-2)

#include "param.h"

// que se passe-t-il en terme de performance et profiling si on echange tx et ty ?
__global__ void heat3d_ftcs_sharedmem_order2_kernel(real_t* A, real_t* B, 
						    int pitch,
						    int imax, int jmax, int kmax)
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
  
  /*
   * rappel : pitch represente la taille en elements de chaque ligne apres
   * "padding" eventuel par cudaMallocPitch pour assurer que chaque debut ligne
   * se trouve a une addresse optimisee pour la coalescence (i.e. un
   * multiple de 16*4=64 bytes, 16 etant la taille d'un demi-warp).
   *
   */
  const int index = /* TODO */;
  
  int low, mid, top, tmp;
  low=0;
  mid=1;
  top=2;

  // load the first 3 planes from global mem to shared mem
  __shared__ real_t shmem[3][BLOCK_DIMY][BLOCK_DIMX/*+1*/];

  if( /* TODO */) 
    {
      shmem[low][ty][tx] = A[/* TODO */];
      shmem[mid][ty][tx] = A[/* TODO */];
      shmem[top][ty][tx] = A[/* TODO */];
    }
    
  __syncthreads();
  

  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=1; k<kmax-2; ++k) {

    if( /* TODO */ )
      {

	B[index+k*pitch*jmax]= 
	  o2Gpu.R3*shmem[mid][ty][tx] + 
	  o2Gpu.R*(shmem[mid][ty-1][tx]+shmem[mid][ty+1][tx]+
		   shmem[mid][ty][tx-1]+shmem[mid][ty][tx+1]+
		   shmem[top][ty][tx]  +shmem[low][ty][tx]);
	
      }
    
    __syncthreads();
    
    // swap planes
    /*
     * TODO
     */


    // update new top data
    if((j>=0) and (j<jmax) and (i>=0) and (i<imax)) 
      {
	shmem[top][ty][tx] = A[ /* TODO */ ];
      }
    
    __syncthreads();
  } // end for (k ...
  
  // last plane
  int k = kmax-2;
  if(tx>0 and tx<BLOCK_DIMX-1 and 
     ty>0 and ty<BLOCK_DIMY-1 and 
     i>0  and i < imax-1 and 
     j>0  and j < jmax-1)
    {
      B[ /* TODO */ ]= 
	o2Gpu.R3*shmem[mid][ty  ][tx  ]+ 
	o2Gpu.R*(shmem[mid][ty-1][tx  ]+shmem[mid][ty+1][tx  ]+
		 shmem[mid][ty  ][tx-1]+shmem[mid][ty  ][tx+1]+
		 shmem[top][ty  ][tx  ]+shmem[low][ty  ][tx  ]);
    }

} // heat3d_ftcs_sharedmem_order2_kernel



//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

#define BLOCK_DIMX2		20
#define BLOCK_DIMY2		16
#define BLOCK_INNER_DIMX2	(BLOCK_DIMX2-4)
#define BLOCK_INNER_DIMY2	(BLOCK_DIMY2-4)

#include "param.h"

/** 
 * CUDA Kernel for 4th order computation.
 * Balance between register and shared memory.
 */
__global__ void heat3d_ftcs_sharedmem_order4_kernel(real_t* A, real_t* B, 
						    int pitch,
						    int imax, int jmax, int kmax)
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
  
  /*
   * rappel : pitch represente la taille en elements de chaque ligne apres
   * "padding" eventuel par cudaMallocPitch pour assurer que chaque debut ligne
   * se trouve a une addresse optimisee pour la coalescence (i.e. un
   * multiple de 16*4=64 bytes, 16 etant la taille d'un demi-warp).
   *
   */
  const int index = pitch*j + i;
  
  int low, mid, top;
  low=0;
  mid=1;
  top=2;

  real_t data_zm2;
  real_t data_zp2;

  // load the 3 planes (z=k-1,k,k+1) from global mem to shared mem
  __shared__ real_t shmem[3][BLOCK_DIMY2][BLOCK_DIMX2+1];

  // load all we need to start computation
  if((j>=0) and (j<jmax) and (i>=0) and (i<imax)) 
    {
      data_zm2           = A[index];              // data@ z=0
      shmem[low][ty][tx] = A[index+  pitch*jmax]; // data@ z=1
      shmem[mid][ty][tx] = A[index+2*pitch*jmax]; // data@ z=2
      shmem[top][ty][tx] = A[index+3*pitch*jmax]; // data@ z=3
      data_zp2           = A[index+4*pitch*jmax]; // data@ z=4
    }
  __syncthreads();
  

  // do FTCS time step update and copy back results to global mem buffer B
  for (int k=2; k<kmax-3; k++) {

    if(tx>1 and tx<BLOCK_DIMX2-2 and 
       ty>1 and ty<BLOCK_DIMY2-2 and 
       i >1 and i < imax-2      and 
       j >1 and j < jmax-2)
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

	// finally, write result back to global memory
	B[index+k*pitch*jmax]= tmp;
      }
    __syncthreads();
    
    // swap planes
    data_zm2           = shmem[low][ty][tx];
    shmem[low][ty][tx] = shmem[mid][ty][tx];
    shmem[mid][ty][tx] = shmem[top][ty][tx];
    shmem[top][ty][tx] = data_zp2;

    // update data_zp2 by loading data at z=k+3
    if((j>=0) and (j<jmax) and (i>=0) and (i<imax)) 
      {
	data_zp2 = A[index+(k+3)*pitch*jmax];
      }
    
    __syncthreads();
  } // end for (k ...
  
  // last plane
  int k = kmax-3;
  if(tx>1 and tx<BLOCK_DIMX2-2 and 
     ty>1 and ty<BLOCK_DIMY2-2 and 
     i >1 and i < imax-2 and 
     j >1 and j < jmax-2)
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
      B[index+k*pitch*jmax] = tmp;
    }

} // heat3d_ftcs_sharedmem_order4_kernel

#endif // _HEAT3D_KERNEL_GPU_SHMEM1_H_
