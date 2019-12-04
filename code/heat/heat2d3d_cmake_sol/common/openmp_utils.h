#ifndef OPENMP_UTILS_H
#define OPENMP_UTILS_H

#ifdef _OPENMP
#include <omp.h>
#endif

void print_openmp_status()
{

#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    printf("Using %d OpenMP threads\n", omp_get_num_threads());
  }
#else
  printf("OpenMP not activated\n");
#endif

} // print_openmp_status

#endif // OPENMP_UTILS
