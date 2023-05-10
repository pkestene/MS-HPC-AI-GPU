/*
 * nvcc -O3 --ptxas-options=-v -gencode=arch=compute_80,code=sm_80 saxpy_thrust_v2.cu -o saxpy_thrust_v2
 *
 */

// see also : http://code.google.com/p/thrust/wiki/QuickStartGuide
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#include "CudaTimer.h"


// define a SAXPY functor
struct saxpy : public thrust::binary_function<float,float,float> {
  float a; 
  
  saxpy(float a) : a(a) {}
  
  __host__ __device__ float operator()(float x, float y)
  {
    return a * x + y; 
  } 
}; 

int main(int argc, char* argv[]) { 

  // vector length
  int N = 1<<22;
  if (argc>1)
    N = atoi(argv[1]);

  // generate random data serially on host vectors
  thrust::host_vector<float> h_x(N), h_y(N);
  std::generate(h_x.begin(), h_x.end(), rand);
  std::generate(h_y.begin(), h_y.end(), rand);

  // copy host to device vectors
  thrust::device_vector<float> d_x = h_x;
  thrust::device_vector<float> d_y = h_y;

  // perform SAXPY
  const float a = 2.0f; 

  CudaTimer gpuTimer;
  gpuTimer.start();

  thrust::transform(d_x.begin(), d_x.end(), 
		    d_y.begin(), 
		    d_y.begin(), 
		    saxpy(a)); 

  gpuTimer.stop();

  // print performance
  double time = gpuTimer.elapsed();
  printf("THRUST SAXPY V2: %8d elements, %10.6f ms per iteration, %6.3f GFLOP/s, %7.3f GB/s\n",
         N,
         time*1000,
         2.0*N / (time*1e9),
         3.0*N*sizeof(float) / (time*1e9) );

  return 1;

}
