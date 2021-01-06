/*
 * How to build:
 *
 * nvcc -arch=sm_80 -o helloworld_array helloworld_array.cu
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

#include "cuda_error.h"

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

  int i = /*  TODO */

  if ( /* TODO */ )
    c[i] = a[i] + b[i];

} // add

/*
 * main
 */
int main( int argc, char* argv[] )
{
  // array size
  int N = 16;

  // host variables
  int *a, *b, *c;

  // device variables
  int *dev_a, *dev_b, *dev_c;
  
  // CPU memory allocation / initialization
  a = (int *) malloc(N*sizeof(int));
  b = (int *) malloc(N*sizeof(int));
  c = (int *) malloc(N*sizeof(int));
  for (int i=0; i<N; i++) {
    a[i]=i;
    b[i]=N-i;
  }

  // GPU device memory allocation / initialization
  CUDA_API_CHECK( cudaMalloc( /* TODO */ ) );

  CUDA_API_CHECK( cudaMemcpy( /* TODO */ );
  
  // perform computation on GPU
  int nbThreadsPerBlock = 8;
  dim3 blockSize(nbThreadsPerBlock,1,1);
  dim3 gridSize(N/nbThreadsPerBlock+1,1,1);
  add<<<gridSize,blockSize>>>( dev_a, dev_b, dev_c, N );
  CUDA_KERNEL_CHECK("add");

  // get back computation result into host CPU memory
  CUDA_API_CHECK( /* TODO */ );

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

  // de-allocate CPU host memory
  free(c);
  free(b);
  free(a);

  // de-allocate GPU device memory
  CUDA_API_CHECK( cudaFree( /* TODO */ ) );

  return EXIT_SUCCESS;
}
