#ifndef MY_CUDA_HELPER_H
#define MY_CUDA_HELPER_H

#include <stdlib.h>
#include "cuda_error.h"

#include "cuda_helper/helper_string.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID) {
  int deviceCount;
  CUDA_API_CHECK( cudaGetDeviceCount(&deviceCount) );

  if (deviceCount == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  
  if (devID < 0) {
    devID = 0;
  }

  if (devID > deviceCount - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            deviceCount);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  cudaDeviceProp deviceProp;
  CUDA_API_CHECK( cudaGetDeviceProperties(&deviceProp, devID) );

  if (deviceProp.major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(-1);
  }

  CUDA_API_CHECK( cudaSetDevice(devID) );
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

  return devID;

} // gpuDeviceInit

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the #
  // of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM
            // minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x10, 8},   // Tesla Generation (SM 1.0) G80 class
      {0x11, 8},   // Tesla Generation (SM 1.1) G8x class
      {0x12, 8},   // Tesla Generation (SM 1.2) G9x class
      {0x13, 8},   // Tesla Generation (SM 1.3) GT200 class
      {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
      {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the "
         "latest SDK)!\n",
         major, minor);
  return -1;
}
// end of GPU Architecture definitions

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&device_count);

  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
    }

    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else {
      sm_per_multiproc =
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc *
                       deviceProp.clockRate;

    if (compute_perf > max_compute_perf) {
      // If we find GPU with SM major > 2, search only these
      if (best_SM_arch > 2) {
        // If our device==dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }

    ++current_device;
  }

  return max_perf_device;
}

// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv) {
  cudaDeviceProp deviceProp;
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(-1);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(-1);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    CUDA_API_CHECK( cudaSetDevice(devID) );
    CUDA_API_CHECK( cudaGetDeviceProperties(&deviceProp, devID) );
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  return devID;
}

#endif // MY_CUDA_HELPER_H
