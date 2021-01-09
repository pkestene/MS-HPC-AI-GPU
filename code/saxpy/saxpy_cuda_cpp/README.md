# Comparing saxpy performance with OpenMP and CUDA implementation 

example adapted from:  
http://www.speedup.ch/workshops/w38_2009/tutorial.html

Example results on platform
GPU : Quadro K2200
CPU : Intel Xeon CPU E5-1620 v3 @ 3.50GHz
cuda toolkit 10.1

OUR CPU CODE:  4194304 elements,   2.378790 ms per iteration,  3.526 GFLOP/s,  21.159 GB/s
OUR GPU CODE:  4194304 elements,   0.804800 ms per iteration, 10.423 GFLOP/s,  62.539 GB/s
CUBLAS      :  4194304 elements,   0.770336 ms per iteration, 10.890 GFLOP/s,  65.337 GB/s


## Activity

- build with e.g. : `make CUDA_ARCH=sm_75`
- run for multiple values of array size, and plot the speed-up of the different parallel implementation versus the serial version.
- Comments ?
