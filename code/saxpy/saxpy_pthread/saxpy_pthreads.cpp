/*
 * g++ -pthread -O3 saxpy_pthreads.cpp -o saxpy_pthreads
 *
 */
#include "SimpleTimer.h"

#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <cstdlib>
#include <cstdio>

// ==================================================
// ==================================================
void saxpy(float* x, float *y, float* z, float a, int n)
{
  std::vector<std::thread> threads;

  int const num_threads = std::thread::hardware_concurrency();

  for(int t = 0; t < num_threads; ++t)
  {
    threads.push_back(std::thread(
                        [=](int thread_no)
                        {
                          for(int i=thread_no; i<n; i+=num_threads)
                          {
                            z[i] = a*x[i]+y[i];
                          }
                        }, t));
  }
  for(std::thread& th : threads)
    th.join();

} // saxpy

// ==================================================
// ==================================================
int main(int argc, char* argv[])
{

  // log2 of vector length
  int log2N = (argc > 1) ? atoi(argv[1]) : 22;
  int N = 1 << log2N;

  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist {0, 100};

  auto gen = [&dist, &mersenne_engine](){
               return dist(mersenne_engine);
             };

  // generate random data
  std::vector<float> x(N), y(N);
  std::generate(x.begin(), x.end(), gen);
  std::generate(y.begin(), y.end(), gen);

  // perform SAXPY
  const float a = 2.0f;

  SimpleTimer aTimer;
  aTimer.start();

  saxpy(x.data(), y.data(), x.data(), a, N);

  aTimer.stop();

  // print performance
  double time = aTimer.elapsedSeconds();
  printf("PTHREADS: %8d elements, %10.6f ms per iteration, %6.3f GFLOP/s, %7.3f GB/s\n",
         N,
         time*1000,
         2.0*N / (time*1e9),
         3.0*N*sizeof(float) / (time*1e9) );

  return 1;

}
