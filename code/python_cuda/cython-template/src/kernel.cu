#include <stdio.h>

template<typename T>
void __global__ kernel_add_one(T* a, int length) {
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    while(gid < length) {
      a[gid] += (T) 1;
        gid += blockDim.x*gridDim.x;
    }
}
