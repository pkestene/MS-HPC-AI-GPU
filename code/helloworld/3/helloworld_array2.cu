/*
 * Please adapt to your actual hardware architecture.
 *
 * nvcc -arch=sm_80 -o helloworld_array2 helloworld_array2.cu -lnvToolsExt
 */

/*
 * adapted from :
 * CUDA by example, by Jason Sanders and Edward Kandrot,
 * Addison-Wesley, 2010
 * http://developer.nvidia.com/cuda-example-introduction-general-purpose-gpu-programming
 */

/* Modified version to handle arrays */


#include <stdio.h>

#include <nvToolsExt.h>

#include "cuda_error.h"

/**
 * a simple CUDA kernel
 *
 * \param[in]  a input array
 * \param[out] b output array
 */
__global__ void compute_gpu( int *a, int *b, int n ) 
{

  int i = threadIdx.x + blockIdx.y * blockDim.x;

  if (i<n)
    b[i] += 2*a[i];

} // compute_gpu

void compute_cpu( int *a, int *b, int n)
{
  nvtxRangePush(__FUNCTION__);

  for (int i=0; i<n; ++i)
    b[i] += 3*a[i];

  nvtxRangePop();

} // compute_cpu

/*
 * main
 */
int main( int argc, char* argv[] ) 
{
  // array size
  int N = 1000000;

  // host variables
  int *a, *b;

  // device variables
  int *dev_a, *dev_b;
  
  // CPU memory allocation / initialization
  a = (int *) malloc(N*sizeof(int));
  b = (int *) malloc(N*sizeof(int));
  for (int i=0; i<N; i++) {
    a[i]=i;
    b[i]=0;
  }

  // GPU device memory allocation / initialization
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_a, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMalloc( (void**)&dev_b, N*sizeof(int) ) );
  CUDA_API_CHECK( cudaMemcpy( dev_a, a, N*sizeof(int),
                              cudaMemcpyHostToDevice ) );
  
  
  // perform computation on GPU
  int nbThreadsPerBlock = 128;
  dim3 blockSize(nbThreadsPerBlock,1,1);
  dim3 gridSize(N/nbThreadsPerBlock+1,1,1);
  compute_gpu<<<gridSize,blockSize>>>( dev_a, dev_b, N );
  CUDA_KERNEL_CHECK("compute_gpu");

  // perform computation on CPU
  compute_cpu(a,b,N);

  // and again on GPU
  compute_gpu<<<gridSize,blockSize>>>( dev_a, dev_b, N );
  CUDA_KERNEL_CHECK("compute_gpu");

  // and again on CPU
  compute_cpu(a,b,N);

  CUDA_API_CHECK( cudaMemcpy( b, dev_b, N*sizeof(int),
                              cudaMemcpyDeviceToHost ) );
  

  // de-allocate CPU host memory
  free(b);
  free(a);

  // de-allocate GPU device memory
  CUDA_API_CHECK( cudaFree( dev_b ) );
  CUDA_API_CHECK( cudaFree( dev_a ) );

  return EXIT_SUCCESS;
}
