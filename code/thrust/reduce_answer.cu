/*
 * Local machine (training room):
 * nvcc -O3 -gencode arch=compute_20,code=sm_20 -o reduce reduce_answer.cu --ptxas-options -v
 *
 * Poincare:
 * nvcc -O3 -gencode arch=compute_35,code=sm_35 -o reduce reduce_answer.cu --ptxas-options -v
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
  GpuTimer timer1;
  timer1.start();
  double sumCpu = thrust::reduce(h_vec.begin(), h_vec.end(), init, binary_op);
  timer1.stop();

  printf("sum cpu %.12f\n",sumCpu);

  printf("Thrust Reduction on CPU, Throughput = %8.4f GB/s, Time = %10.5f ms, Size = %u Elements\n", 1.0e-9 * ((double)N*sizeof(double))/(timer1.elapsed()/1000), timer1.elapsed(), N);

  // transfer data to the device
  thrust::device_vector<double> d_vec = h_vec;

  // reset initial value of the reduction before computing on GPU
  init = 0.0;

  /*
   * reduce data on the device
   */
  GpuTimer timer2;
  timer2.start();
  double sumGpu = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
  timer2.stop();

  printf("sum gpu %.12f\n",sumGpu);

  printf("Thrust Reduction on GPU, Throughput = %8.4f GB/s, Time = %10.5f ms, Size = %u Elements\n", 1.0e-9 * ((double)N*sizeof(double))/(timer2.elapsed()/1000), timer2.elapsed(), N);

  return 0;
}
