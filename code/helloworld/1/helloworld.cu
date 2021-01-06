// nvcc helloworld.cu -o helloworld

#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void) {
  printf("Hello World from GPU thread %d, block %d !\n",
         threadIdx.x, blockIdx.x);
}

int main(int argc, char* argv[]) {
  printf("Hello from CPU !\n");
  print_from_gpu<<<1,1>>>();
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}
