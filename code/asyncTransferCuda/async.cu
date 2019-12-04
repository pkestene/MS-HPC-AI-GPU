/*
 *  Copyright 2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
// translated to Cuda/C by P. Kestener

// include system
#include <stdlib.h>
#include <stdio.h>

typedef float real_t;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// utility
double maxval(real_t *a, int size)
{
  double res=0;
  for (int i=0; i<size; i++) {
    res = fmax(res, fabs(a[i]-1.0));
  }

  return res;
}

// CUDA kernel
__global__ void kernel(real_t *a, int offset)
{
  int i;
  real_t c, s, x;
    
  i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  x = i; s = sin(x); c = cos(x);
  a[i] = a[i] + sqrt(s*s+c*c);
}

int main(int arc, char* argv[])
{

  const int blockSize = 256, nStreams = 4;
  const int n = 4*1024*blockSize*nStreams;

  // host array
  real_t *h_a;

  // device array
  real_t *d_a;

  // streams
  cudaStream_t stream[nStreams];
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  float time;
  int offset, streamSize = n/nStreams;
  int streamSizeBytes = streamSize*sizeof(real_t);
  cudaDeviceProp prop;
  
  // get device/gpu properties
  checkCuda ( cudaGetDeviceProperties(&prop, 0) );
  printf("Device: %s\n", prop.name);

  // allocate pinned  host memory
  checkCuda( cudaMallocHost((void**)&h_a, n*sizeof(real_t)) );

  // allocate device memory
  checkCuda( cudaMalloc((void**)&d_a, n*sizeof(real_t)) );

  // create events and streams
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) ); 
  for (int i = 0; i < nStreams; i++) {
    checkCuda( cudaStreamCreate(&(stream[i]) ) );
  }
  
  /*
   * baseline case - sequential transfer and execute
   */
  for (int i=0; i<n; i++) h_a[i]=0;
  checkCuda ( cudaEventRecord(startEvent,0) );

  checkCuda( cudaMemcpy(d_a, h_a, n*sizeof(real_t), cudaMemcpyHostToDevice) );
  kernel<<<n/blockSize, blockSize>>>(d_a, 0);
  checkCuda( cudaMemcpy(h_a, d_a, n*sizeof(real_t), cudaMemcpyDeviceToHost) );

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf( "Time for sequential transfer and execute (ms): %f\n", time);
  printf( "  max error: %g\n", maxval(h_a,n) );

  /*
   * asynchronous version 1: loop over {copy, kernel, copy}
   */
  for (int i=0; i<n; i++) h_a[i]=0;
  checkCuda( cudaEventRecord(startEvent,0) );

  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    checkCuda( cudaMemcpyAsync(d_a+offset,h_a+offset,streamSizeBytes,cudaMemcpyHostToDevice,stream[i]) );
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a,offset);
    checkCuda( cudaMemcpyAsync(h_a+offset,d_a+offset,streamSizeBytes,cudaMemcpyDeviceToHost,stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf( "Time for asynchronous V1 transfer and execute (ms): %f\n", time);
  printf( "  max error: %g\n", maxval(h_a,n) );
  
  /*
   * asynchronous version 2: 
   * loop over copy, loop over kernel, loop over copy
   */
  for (int i=0; i<n; i++) h_a[i]=0;
  checkCuda( cudaEventRecord(startEvent,0) );
  
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    checkCuda( cudaMemcpyAsync(d_a+offset,h_a+offset,streamSizeBytes,cudaMemcpyHostToDevice,stream[i]) );
  }
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    kernel<<<streamSize/blockSize, blockSize,
      0, stream[i]>>>(d_a,offset);
  }
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    checkCuda( cudaMemcpyAsync(h_a+offset,d_a+offset,streamSizeBytes,cudaMemcpyDeviceToHost,stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf( "Time for asynchronous V2 transfer and execute (ms): %f\n", time );
  printf( "  max error: %g\n", maxval(h_a,n) );

  /*
   * asynchronous version 3: 
   * loop over copy, loop over {kernel, event}, loop over copy
   */
  for (int i=0; i<n; i++) h_a[i]=0;
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    checkCuda( cudaMemcpyAsync(d_a+offset,h_a+offset,streamSizeBytes,cudaMemcpyHostToDevice,stream[i]) );
  }
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a,offset);
    checkCuda( cudaEventRecord(dummyEvent, stream[i]) );
  }
  for (int i = 0; i < nStreams; i++) {
    offset = i*streamSize;
    checkCuda( cudaMemcpyAsync(h_a+offset,d_a+offset,streamSizeBytes,cudaMemcpyDeviceToHost,stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf( "Time for asynchronous V3 transfer and execute (ms): %f\n", time );
  printf( "  max error: %g\n", maxval(h_a,n) );
  
  // cleanup
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cudaEventDestroy(dummyEvent);
  for (int i=0; i<nStreams; i++) {
    checkCuda( cudaStreamDestroy(stream[i]) );
  }
  cudaFreeHost(h_a);
  cudaFree(d_a);

  return 0;
}

