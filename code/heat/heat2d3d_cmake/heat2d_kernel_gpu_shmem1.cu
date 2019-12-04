/**
 * \file heat2d_kernel_gpu_shmem1.cu
 * \brief Implementation of CUDA kernel to solve 2D heat equation.
 *
 * \author Pierre Kestener
 * \date 18-dec-2009
 */

#ifndef _HEAT2D_KERNEL_GPU_SHMEM1_H_
#define _HEAT2D_KERNEL_GPU_SHMEM1_H_

#include <stdio.h> // for printf

#include "param.h" // for real_t typedef

#define BLOCK_DIMX		16
#define BLOCK_DIMY		16
#define BLOCK_INNER_DIMX	(BLOCK_DIMX-2)
#define BLOCK_INNER_DIMY	(BLOCK_DIMY-2)


/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory 
 */
// que se passe-t-il en terme de performance et profiling si on echange tx et ty ?
__global__ void heat2d_ftcs_sharedmem_order2_kernel(real_t* A, real_t* B, 
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
  
  /*
   * rappel : pitch represente la taille en elements de chaque ligne apres
   * "padding" eventuel par cudaMallocPitch pour assurer que chaque debut ligne
   * se trouve a une addresse optimisee pour la coalescence (i.e. un
   * multiple de 16*4=64 bytes, 16 etant la taille d'un demi-warp).
   *
   */
  const int index = /* TODO*/;
  
  // copy data from global mem to shared mem (take care of bank conflicts)
  __shared__ real_t shmem[/* TODO */][/* TODO */];
  if( /* TODO */ ) 
    {
      shmem[ty][tx] = A[index];
    }
    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if( /* TODO */ )
    {
      B[index]= o2Gpu.R2*shmem[ty][tx] + 
	o2Gpu.R*(shmem[ty-1][tx]+shmem[ty+1][tx])+
	o2Gpu.R*(shmem[ty][tx-1]+shmem[ty][tx+1]);
    }

} // heat2d_ftcs_sharedmem_order2_kernel

/**
 * CUDA kernel for 2nd order 2D heat solver using GPU shared memory
 * and using a different stencil.
 */
__global__ void heat2d_ftcs_sharedmem_order2b_kernel(real_t* A, real_t* B, 
						     int pitch, int imax, int jmax)
{

  /* TODO */

} // heat2d_ftcs_sharedmem_order2b_kernel

#define BLOCK_DIMX2		20
#define BLOCK_DIMY2		16
#define BLOCK_INNER_DIMX2	/*TODO*/
#define BLOCK_INNER_DIMY2	/*TODO*/


/**
 * CUDA kernel for 4th order 2D heat solver using GPU shared memory 
 */
__global__ void heat2d_ftcs_sharedmem_order4_kernel(real_t* A, real_t* B, 
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
  
  /*
   * rappel : pitch represente la taille en elements de chaque ligne apres
   * "padding" eventuel par cudaMallocPitch pour assurer que chaque debut ligne
   * se trouve a une addresse optimisee pour la coalescence (i.e. un
   * multiple de 16*4=64 bytes, 16 etant la taille d'un demi-warp).
   *
   */
  const int index = pitch*j + i;
  
  // copy data from global mem to shared mem (take care of bank conflicts)
  __shared__ real_t shmem[/* TODO */][/* TODO */];
  if((j>=0) and (j<jmax) and (i>=0) and (i<imax)) 
    {
      shmem[ty][tx] = A[index];
    }
    
  __syncthreads();
  
  // do FTCS time step update and copy back results to global mem buffer B
  if( /* TODO */ )
    {
      B[index]= o4Gpu.S2*shmem[ty][tx] + 
	o4Gpu.S*(-shmem[ty-2][tx]+16*shmem[ty-1][tx]+16*shmem[ty+1][tx]-shmem[ty+2][tx])+
	o4Gpu.S*(-shmem[ty][tx-2]+16*shmem[ty][tx-1]+16*shmem[ty][tx+1]-shmem[ty][tx+2]);
    }

} // heat2d_ftcs_sharedmem_order4_kernel

#endif // _HEAT2D_KERNEL_GPU_SHMEM1_H_
