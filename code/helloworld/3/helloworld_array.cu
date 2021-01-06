/*
 * Please adapt to your actual hardware architecture.
 *
 * nvcc -arch=sm_80 --ptxas-options -v -o helloworld_array helloworld_array.cu
 */

/*
 * adapted from :
 * CUDA by example, by Jason Sanders and Edward Kandrot,
 * Addison-Wesley, 2010
 * http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
 */

/* Modified version to handle arrays */


#include <stdio.h>

#include "cuda_error.h"

/**
 * a simple CUDA kernel
 *
 * \param[in]  a input array
 * \param[in]  b input array
 * \param[out] c output array
 */
__global__ void add( int *a, int *b, int *c, int n ) 
{

  int i = threadIdx.x + blockIdx.y * blockDim.x;

  if (i<n)
    c[i] = a[i] + b[i];

} // add

/*
 * main
 */
int main( int argc, char* argv[] ) 
{
  // array size
  int N = 1000000;

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
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_a, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_b, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_c, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMemcpy( dev_a, a, N*sizeof(int),
                              cudaMemcpyHostToDevice ) );
  CUDA_API_CHECK( cudaMemcpy( dev_b, b, N*sizeof(int),
                              cudaMemcpyHostToDevice ) );
  
  
  // perform computation on GPU
  int nbThreadsPerBlock = 1280;
  dim3 blockSize(nbThreadsPerBlock,1,1);
  dim3 gridSize(N/nbThreadsPerBlock+1,1,1);
  add<<<gridSize,blockSize>>>( dev_a, dev_b, dev_c, N );
  CUDA_KERNEL_CHECK("add");

  // get back computation result into host CPU memory
  CUDA_API_CHECK( cudaMemcpy( c, dev_c, N*sizeof(int),
                              cudaMemcpyDeviceToHost ) );

  // output result on screen
  int passed=1;
  for (int i=0; i<N; i++) {
    if (c[i] != N) {
      passed = 0;
      //printf("wrong value : %d %d\n",i,c[i]);
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
  CUDA_API_CHECK( cudaFree( dev_c ) );
  CUDA_API_CHECK( cudaFree( dev_b ) );
  CUDA_API_CHECK( cudaFree( dev_a ) );

  return EXIT_SUCCESS;
}
