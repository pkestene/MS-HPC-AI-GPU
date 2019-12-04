# Additional information

- http://www.pgroup.com/lit/articles/insider/v3n1a4.htm
- http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
- https://github.com/parallel-forall/code-samples/blob/master/series/cuda-fortran/async.cuf

# How to build the cuda/fortran version

Cuda/Fortran version for mdlslx83 (GPU K80 with architecture sm_37):
async.cuf

```bash
pgfortran -Mcuda=7.5 -Mcuda=cc35 -Mcuda=ptxinfo -o async_cuf async.cuf
```

# How to build the Cuda/C++ version

async.cu

```bash
nvcc -gencode arch=compute_35,code=sm_35 --ptxas-options -v -o async_cu async.cu
```
