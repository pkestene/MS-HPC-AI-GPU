#ifndef OPENMP_UTILS_H
#define OPENMP_UTILS_H

#include <iostream> // std::cout
#include <cstdlib> // for std::getenv
#include <omp.h>

void print_openmp_version(void) {

#ifdef _OPENMP
  int my_omp_v = _OPENMP;

  // thread id
  int omp_tid;

  // number of threads in parallel region
  // if OMP_NUM_THREADS is defined in environment then
  // nbThreads = OMP_NUM_THREADS
  // else
  // nbThreads is defined by OpenMP runtime
  int nbThreads;

  // number of threads requested by user
  const char* omp_num_threads =
    std::getenv("OMP_NUM_THREADS") == NULL ? 
    "Undefined" : std::getenv("OMP_NUM_THREADS");

#pragma omp parallel private(omp_tid) shared(nbThreads)
  {
    // get current thread id
    omp_tid = omp_get_thread_num();
#pragma omp master
    {
      nbThreads = omp_get_num_threads();
      std::cout << "OPENMP version: " << my_omp_v << "\n"
                << "number of OpenMP threads=" << nbThreads << "\n"
                << "OMP_NUM_THREADS=" << omp_num_threads << "\n";
      std::cout << std::endl;
    }
  } // end parallel region

#endif // _OPENMP

} // print_openmp_version

#endif // OPENMP_UTILS
