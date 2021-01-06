/*
 * How to build:
 *
 * nvcc -arch=sm_80 -o helloworld_array_managed helloworld_array_managed.cu
 */

/*
 * Originally from the book, CUDA by example, 
 * by Jason Sanders and Edward Kandrot,
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

#include <stdio.h>

#include "../cuda_error.h"

/**
 * a simple CUDA kernel to add two arrays
 *
 * \param[in]  a input array
 * \param[in]  b input array
 * \param[out] c output array
 * \param[in]  n array size
 */
__global__ void add( int *a, int *b, int *c, int n )
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i<n)
    c[i] = a[i] + b[i];

} // add

/*
 * main
 */
int main( int argc, char* argv[] )
{
  // array size
  int N = 16;

  // host and device variables
  int *a, *b, *c;

  // CPU / GPU memory allocation
  CUDA_API_CHECK( cudaMallocManaged( &a, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMallocManaged( &b, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMallocManaged( &c, N*sizeof(int) ) );
    
  // CPU / GPU memory initialization
  for (int i=0; i<N; i++)
  {
    a[i] = i;
    b[i] = N-i;
    c[i] = 0;
  }
  
  // perform computation on GPU
  int nbThreadsPerBlock = 8;
  dim3 blockSize(nbThreadsPerBlock,1,1);
  dim3 gridSize(N/nbThreadsPerBlock+1,1,1);
  add<<<gridSize,blockSize>>>( a, b, c, N );
  CUDA_KERNEL_CHECK("add");

  // Wait for GPU to finish before accessing array c on host
  cudaDeviceSynchronize();

  // output result on screen
  int passed=1;
  for (int i=0; i<N; i++) {
    if (c[i] != N) {
      passed = 0;
      printf("wrong value : %d %d\n",i,c[i]);
    }
  }
  if (passed) {
    printf("test succeeded !\n");
  } else {
    printf("test failed !\n");
  }

  // de-allocate CPU / GPU device memory
  CUDA_API_CHECK( cudaFree( c ) );
  CUDA_API_CHECK( cudaFree( b ) );
  CUDA_API_CHECK( cudaFree( a ) );

  return EXIT_SUCCESS;
}
