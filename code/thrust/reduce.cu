/*
 * nvcc -O3 -gencode arch=compute_30,code=sm_30 -o reduce reduce.cu --ptxas-options -v
 */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include <cstdlib>
#include <cstdio> // for printf

#include "GpuTimer.h"
#include "CpuTimer.h"

double my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

int main(void)
{

  // binary operation used to reduce values
  thrust::plus<double> binary_op;

  // generate N random numbers serially
  int N = 1<<24;
  thrust::host_vector<double> h_vec(N);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  // initial value of the reduction
  double init = 0.0;

  /*
   * reduce data on host
   */
  /* TODO : add time measurement */
  double sumCpu = thrust::reduce(h_vec.begin(), h_vec.end(), init, binary_op);

  printf("sum cpu %.12f\n",sumCpu);

  /* TODO : print performance in GBytes/s */

  // transfer data to the device
  thrust::device_vector<double> d_vec = h_vec;

  // reset initial value of the reduction before computing on GPU
  init = 0.0;

  /*
   * reduce data on the device
   */
  /* TODO : add time measurement */
  double sumGpu = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);

  printf("sum gpu %.12f\n",sumGpu);

  /* TODO : print performance in GBytes/s */

  return 0;
}
