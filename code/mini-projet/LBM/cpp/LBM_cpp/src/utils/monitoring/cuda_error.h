/**
 * CUDA error checking utilities
 *
 * More complete error checking macro are 
 * available in the CUDA samples:
 * /usr/local/cuda-10.1/samples/common/inc/helper_cuda.h
 * These macros are more generic since they can
 * deal with CUDA core API, as well as all other
 * CUDA libraries (cufft, curand, ...)
 *
 * See also https://codingbyexample.com/2019/02/06/error-handling-for-gpu-cuda/
 */
#pragma once

#include <cuda.h>
#include <string>
#include <iostream>

#include <cublas_v2.h>

// if symbol ALWAYS_SYNC_GPU is defined in your build system
// then we will always call cudaDeviceSynchronize, before checking
// last error from a cuda kernel call
#ifdef ALWAYS_SYNC_GPU
#define FORCE_SYNC_GPU 1
#else
#define FORCE_SYNC_GPU 0
#endif


/**
 * Preprocessor macro helping to retrieve the exact code 
 * location where the error was emitted.
 */
#define CUDA_API_CHECK(value) cuda_api_check((value), #value, __FILE__, __LINE__)

 /**
 * Check CUDA API call status (e.g. cudaMemcpy for memory allocation)
 * see https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
 */
static void cuda_api_check(cudaError_t       status, 
                           const char *const func, 
                           const char *const file,
                           const int         line) 
{

  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA API error at %s:%d code=%d(%s) \"%s\" \n", 
            file, line,
            static_cast<unsigned int>(status), 
            cudaGetErrorName(status), func);

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }

} // cuda_api_check

/**
 * cuBLAS API errors
 * taken from Nvidia Cuda samples helper_cuda.h
 */
static const char *cublasGetErrorString(cublasStatus_t error) 
{
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";

} // cublasGetErrorString

/**
 * Check CUBLAS API call status (e.g. cublasSaxpy)
 * see https://docs.nvidia.com/cuda/cublas/index.html
 */
static void cublas_api_check(cublasStatus_t    status, 
                             const char *const func, 
                             const char *const file,
                             const int         line) 
{

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS API error at %s:%d code=%d(%s) \"%s\" \n", 
            file, line,
            static_cast<unsigned int>(status), 
            cublasGetErrorString(status), func);

    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

/**
 * enum used below; can be used as the second argument of macro 
 * CUDA_KERNEL_CHECK
 */
enum device_sync_t {
  DEVICE_NO_SYNC = 0,
  DEVICE_SYNC = 1
};


/**
 * a simple macro helper:
 * GET_KERNEL_CHECK_MACRO always picks the 3rd arg
 *
 * see https://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
 */
#define GET_KERNEL_CHECK_MACRO(_1,_2,NAME,...) NAME

/**
 * another simple macro helper :
 * - if CUDA_KERNEL_CHECK is called with only 1 argument, then CUDA_KERNEL_CHECK1 is chosen
 * - if CUDA_KERNEL_CHECK is called with      2 arguments, then CUDA_KERNEL_CHECK2 is chosen
 *
 *
 * this is the macro we want to call
 */
#define CUDA_KERNEL_CHECK(...) GET_KERNEL_CHECK_MACRO(__VA_ARGS__, CUDA_KERNEL_CHECK2, CUDA_KERNEL_CHECK1)(__VA_ARGS__)

/**
 * Preprocessor macro helping to retrieve the exact code 
 * location where the error was emitted.
 *
 * Default behavior, don't synchronize device
 */
#define CUDA_KERNEL_CHECK1(msg) cuda_kernel_check((msg), __FILE__, __LINE__, DEVICE_NO_SYNC)

/**
 * Same as above, but let the user chose if we want to synchronize device.
 */
#define CUDA_KERNEL_CHECK2(msg,sync) cuda_kernel_check((msg), __FILE__, __LINE__, sync)

/**
 * Check last CUDA kernel call status.
 * If it was not successfull then print error message.
 *
 * \param[in] errstr error message to print
 * \param[in] file source filename where error occured
 * \param[in] line line number where error occured
 * \param[in] sync integer, 0 means no device synchronization
 */
static void cuda_kernel_check(const char* errstr,
                              const char* file,
                              const int   line,
                              const int   sync)
{

  if (sync or FORCE_SYNC_GPU) {
    //fprintf(stderr, "syncing device\n");
    cudaDeviceSynchronize();
  }

  auto status = cudaGetLastError();

  if (status != cudaSuccess) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errstr, static_cast<int>(status),
            cudaGetErrorString(status));
    
    //cudaDeviceReset();
    //exit(EXIT_FAILURE);
  }

} // cuda_kernel_check

