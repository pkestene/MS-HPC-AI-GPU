/*
 * nvcc -O3 --ptxas-options -v -o bandwidth bandwidth.cu
 */

#include <stdio.h>
#include <stdlib.h>

#include "CudaTimer.h"

/*
 * handle CUDA-related error messages
 *
 * Place a call to
 * HANDLE_ERROR( "Some kernel" );
 * right after an actual cuda kernel call
 *
 */
static void HandleError( const char* kernelName,
                         const char *file,
                         int line ) {
  cudaError_t err = cudaPeekAtLastError();

  if (err != cudaSuccess) {
    printf("Kernel %s FAILED in %s at line %d with error message:\n%s\n",
	   kernelName,
	   file,
	   line,
	   cudaGetErrorString(cudaGetLastError()));
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( kernelName ) (HandleError( kernelName, __FILE__, __LINE__ ))


/**
 * a simple CUDA kernel
 *
 * \param[in]  a input array
 * \param[out] b output array
 */
__global__ void copy( int *a, int *b, int n ) {

  /* TODO */

}

/**
 * a simple CUDA kernel
 *
 * \param[in]  a input array
 * \param[out] b output array
 */
__global__ void copy2( int *a, int *b, int n ) {

  /* TODO */

}

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

  // read N from command ligne
  /* TODO : utiliser la routine atoi pour convertir le premier argument en entier */

  printf("Number of bytes to copy: %ld\n",2*N*sizeof(int));

  // CPU memory allocation / initialization
  a = (int *) malloc(N*sizeof(int));
  b = (int *) malloc(N*sizeof(int));
  for (int i=0; i<N; i++) {
    a[i]=rand();
    b[i]=0;
  }

  // GPU device memory allocation / initialization
  cudaMalloc( (void**)&dev_a, N*sizeof(int) );
  cudaMalloc( (void**)&dev_b, N*sizeof(int) );

  // upload a into dev_a
  cudaMemcpy( dev_a, a, N*sizeof(int),
	      cudaMemcpyHostToDevice );


  // create a timer
  CudaTimer copyTimer;

  // start time measurement
  copyTimer.start();

  // version 1
  if (1) {
    int nbThreads = 128;
    dim3 blockSize(nbThreads,1,1);
    dim3 gridSize(N/nbThreads+1,1,1);

    copy<<<gridSize,blockSize>>>( dev_a, dev_b, N );
    HANDLE_ERROR( "copy" );

  }

  // version 2
  if (0) { // disabled
    int nbThreads = 256;

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;

    /* TODO : fill deviceProp by calling cudaGetDeviceProp */

    int nMultiproc = deviceProp.multiProcessorCount;
    dim3 blockSize(nbThreads,1,1);
    dim3 gridSize(100*nMultiproc,1,1);
    copy2<<<gridSize,blockSize>>>( dev_a, dev_b, N );
    HANDLE_ERROR( "copy2" );
  }

  // version 3 (from cuda runtime programing interface)
  if (0) {
    // TODO : just using cudaMemcpy
  }

  copyTimer.stop();


  // get back computation result into host CPU memory
  cudaMemcpy( b, dev_b, N*sizeof(int),
	      cudaMemcpyDeviceToHost );

  // compare results on CPU
  int passed=1;
  for (int i=0; i<N; i++) {
    if (b[i] != a[i]) {
      passed = 0;
      printf("wrong value : %d %d\n",a[i],b[i]);
    }
  }
  if (passed) {
    printf("test succeeded !\n");
  } else {
    printf("test failed !\n");
  }

  // print bandwidth:
  {
    int numBytes = 2*N*sizeof(int); // factor 2 because 1 read + 1 write
    printf("bandwidth is %f GBytes (%f)/s\n",
	   1e-9*numBytes/copyTimer.elapsed_in_second(),
	   copyTimer.elapsed_in_second() );
  }

  // print peak bandwidth
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("  Peak Memory Bandwidth (GB/s): %f\n",
	   2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
  }

  // de-allocate CPU host memory
  free(b);
  free(a);

  // de-allocate GPU device memory
  cudaFree( dev_b );
  cudaFree( dev_a );

  return 0;
}
