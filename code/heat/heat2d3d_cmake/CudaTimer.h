/**
 * \file CudaTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 * \author Pierre Kestener
 * \date 30 Oct 2010
 *
 * $Id: CudaTimer.h 1783 2012-02-21 10:20:07Z pkestene $
 */
#ifndef CUDA_TIMER_H_
#define CUDA_TIMER_H_

//#include <cuda_runtime.h>
#include "cuda_error.h"

/**
 * \brief a simple timer for CUDA kernel.
 */
class CudaTimer
{
protected:
  cudaEvent_t startEv, stopEv;
  double total_time;

public:
  CudaTimer() {
    CUDA_API_CHECK( cudaEventCreate(&startEv) );
    CUDA_API_CHECK( cudaEventCreate(&stopEv) );
    total_time = 0.0;
  }

  ~CudaTimer() {
    CUDA_API_CHECK( cudaEventDestroy(startEv) );
    CUDA_API_CHECK( cudaEventDestroy(stopEv) );
  }

  void start() {
    CUDA_API_CHECK( cudaEventRecord(startEv, 0) );
  }

  void reset() {
    total_time = 0.0;
  }

  /** stop timer and accumulate time in seconds */
  void stop() {
    float gpuTime;
    CUDA_API_CHECK( cudaEventRecord(stopEv, 0) );
    CUDA_API_CHECK( cudaEventSynchronize(stopEv) );
    CUDA_API_CHECK( cudaEventElapsedTime(&gpuTime, startEv, stopEv) );
    total_time += (double)1e-3*gpuTime;
  }
    
  /** return elapsed time in seconds (as record in total_time) */
  double elapsed() {
    return total_time;
  }

}; // class CudaTimer
  
#endif // CUDA_TIMER_H_
