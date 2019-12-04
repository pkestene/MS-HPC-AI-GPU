/*
 * https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels
 *
 * nvcc -arch=sm_50 test.cu -o test
 */

#include<stdlib.h>
#include<iostream>

const size_t size_w = 1024;
const size_t size_h = 1024;

typedef unsigned mytype;
typedef mytype  arr_t[size_w];
const mytype A_val = 1;
const mytype B_val = 2;

__global__ void matrix_add_2D(const arr_t * __restrict__ A, 
                              const arr_t * __restrict__ B, 
                              arr_t * __restrict__ C, 
                              const size_t sw, 
                              const size_t sh){

  size_t idx = threadIdx.x+blockDim.x*(size_t)blockIdx.x;
  size_t idy = threadIdx.y+blockDim.y*(size_t)blockIdx.y;

  if ((idx < sh) && (idy < sw)) C[idx][idy] = A[idx][idy] + B[idx][idy];
}

int main(int argc, char* argv[])
{

  arr_t *A,*B,*C;
  size_t ds = size_w*size_h*sizeof(mytype);
  cudaError_t err = cudaMallocManaged(&A, ds);
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; 
    return 0;
  }

  cudaMallocManaged(&B, ds);
  cudaMallocManaged(&C, ds);
  
  for (int x = 0; x < size_h; x++)
    for (int y = 0; y < size_w; y++) {
      A[x][y] = A_val;
      B[x][y] = B_val;
      C[x][y] = 0;
    }

  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess,0);
  if (attr){
    cudaMemPrefetchAsync(A, ds, 0);
    cudaMemPrefetchAsync(B, ds, 0);
    cudaMemPrefetchAsync(C, ds, 0);
  }

  dim3 threads(32,32);
  dim3 blocks((size_w+threads.x-1)/threads.x, (size_h+threads.y-1)/threads.y);
  
  matrix_add_2D<<<blocks,threads>>>(A,B,C, size_w, size_h);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  
  for (int x = 0; x < size_h; x++)
    for (int y = 0; y < size_w; y++)
      if (C[x][y] != A_val+B_val) {
        std::cout << "mismatch at: " << x << "," << y 
                  << " was: " << C[x][y] 
                  << " should be: " << A_val+B_val
                  << std::endl; 
        return 0;
      }

  std::cout << "Success!" << std::endl;
  
  return 0;
}
