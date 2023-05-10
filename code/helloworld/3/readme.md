# Testing error handling for cuda computing.

## Fixing errors

helloworld_array is broken, can you fix it ?

## Sync or async ?

Here we revisit again a helloworld (adding two vector of integers), and visualize timeline.

- build helloword_array2
  ```shell
  nvcc -g -G -arch=sm_XX -o helloworld_array2 helloworld_array2.cu -lnvToolsExt
  ```
- launch `nsys-ui` (Nvidia visual profiling)

Manipulations:
- Can you interpret the visual trace (look at cpu / gpu trace) ?

- Open source code, and re-order computation calls: first the two calls to compute_gpu, then the calls to compute_cpu. Rebuild. Can can you observe ? What does it illustrate ?


## How to acquire profiling information

- acquire profiling data
  ```shell
  nsys profile --trace=cuda,nvtx --stats=true --output helloworld_array2 ./helloworld_array2
  ```
- visualize profiling data
  ```shell
  nsys-ui ./helloworld_array2.nsys-rep
  ```
