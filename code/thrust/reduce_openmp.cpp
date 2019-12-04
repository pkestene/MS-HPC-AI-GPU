/*
 * g++ -std=c++11 -O3 -I/usr/local/cuda-7.5/include -o reduce_openmp reduce_openmp.cpp CpuTimerOmp.cpp -fopenmp -lgomp
 */

#include <thrust/host_vector.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>

#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/copy.h>

#include <cstdlib>
#include <cstdio> // for printf
#include <random> // for std::default_random_engine, uniform_real_distribution

#include "CpuTimerOmp.h"

double my_rand(void)
{
  //static thrust::default_random_engine rng;
  //static thrust::uniform_real_distribution<double> dist(0.0, 1.0);
  static std::default_random_engine rng;
  static std::uniform_real_distribution<double> dist(0.0, 1.0);
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

  // copy host_vector to openmp vector
  thrust::omp::vector<double> h_vec_omp(N);
  // Don't do this (initialize openmp vector with random value like this !)
  //thrust::generate(thrust::omp::par, h_vec_omp.begin(), h_vec_omp.end(), my_rand);
  h_vec_omp = h_vec;


  // initial value of the reduction
  double init = 0.0;

  /*
   * reduce data on host
   */
  CpuTimerOmp timer1;
  timer1.start();
  double sumCpu = thrust::reduce(thrust::omp::par, h_vec_omp.begin(), h_vec_omp.end(), init, binary_op);
  timer1.stop();

  printf("sum cpu (openmp) %.12f\n",sumCpu);

  printf("Thrust Reduction on CPU, Throughput = %8.4f GB/s, Time = %10.5f ms, Size = %u Elements\n", 1.0e-9 * ((double)N*sizeof(double))/(timer1.elapsed()/1000), timer1.elapsed(), N);

  return 0;
}
