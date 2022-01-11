// collection of useful routines

#include "my_cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas.h>

// ==================================
// ==================================
/* 
 * initialises CUDA and directs all
 * computations to the given
 * CUDA device
 */
void initCuda(const int selectedDevice)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "Sorry, no CUDA device fount");
    exit(1);
  }
  if (selectedDevice >= deviceCount)
  {
    fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount-1);
    exit(2);
  }
  cudaSetDevice(selectedDevice);
  checkErrors("initCuda");

  cublasInit();

} // initCuda

// ==================================
// error checking 
// ==================================
void checkErrors(const char *label)
{

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

} // checkErrors
