/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/*
 * CUDA by example, by Jason Sanders and Edward Kandrot,
 * Addison-Wesley, 2010
 * http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
 */

/*
 * Slightly modified for better error handling : 
 * macro CUDA_KERNEL_CHECK accepts 2 arguments, 
 * first one is a string, 
 * second one cant be DEVICE_SYNC or DEVICE_NO_SYNC to trigger calling
 * cudaDeviceSynchronize before actaully probing the GPU last error.
 *
 * cudaDeviceSynchronize ensures the CPU waits the GPU to finish computing
 * the last kernel, and then we check the status.
 *
 * without GPU synchronization, since GPU computing is asynchronous, 
 * in case of an actual error, we may catch the error message much later
 *
 * with GPU synchronization, we make sure that in case of an actual error,
 * we will catch the error rightaway, but most of the time the kernel 
 * which are running fine, are forced to synchronize, which is not a good
 * idea for performance
 *
 * Finally, while debugging, you might consider defining symbol
 * ALWAYS_SYNC_GPU, is that case all CUDA kernel will be checked with
 * synchronization on.
 *
 * For production run, you can disable synchronization, i.e. just don't 
 * define symbol ALWAYS_SYNC_GPU.
 * 
 */

/*
 * How to build:
 *
 * nvcc -Xcompiler -Wall  -arch=sm_50 -o helloworld helloworld.cu -lcudart
 */

#include <stdio.h>

#include "cuda_error.h"

/**
 * a simple CUDA kernel adding two scalar values
 *
 * \param[in]  a input integer
 * \param[in]  b input integer
 * \param[out] c pointer-to-integer for result
 */
__global__ void add( int a, int b, int *c ) {
  *c = a + b;
}

/*
 * main
 */
int main( int argc, char* argv[] )
{
  int c;
  int *dev_c;
  
  // GPU device memory allocation
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

  // perform computation on GPU
  add<<<1,1>>>( 2, 7, dev_c );
  CUDA_KERNEL_CHECK("add");

  // get back computation result into host CPU memory
  CUDA_API_CHECK( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );

  // output result on screen
  printf( "2 + 7 = %d\n", c );

  // de-allocate GPU device memory
  CUDA_API_CHECK( cudaFree( dev_c ) );

  return EXIT_SUCCESS;
}
