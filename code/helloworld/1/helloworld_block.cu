/*
 * How to build:
 *
 * nvcc -arch=sm_30 -o helloworld_block helloworld_block.cu
 *
 * Note that you need to adjust the architecture version to your current GPU hardware.
 * Hardware version can be probed with e.g. deviceQuery example (from Nvidia SDK samples).
 *
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void hello()
{
  printf("Hello world! I'm a thread %d in block %d\n", threadIdx.x, blockIdx.x);
}


int main(int argc,char **argv)
{

  // default values for 
  // - the number of blocks
  // - the size of the blocks (number of threads per block)
  int numBlocks = 16;
  int blockWidth = 1;

  if (argc>1)
    numBlocks = atoi(argv[1]);
  if (argc>2)
    blockWidth = atoi(argv[2]);

  // launch the kernel
  hello<<<numBlocks, blockWidth>>>();
  
  // force the printf()s to flush
  cudaDeviceSynchronize();
  
  printf("That's all!\n");
  
  return EXIT_SUCCESS;
}
