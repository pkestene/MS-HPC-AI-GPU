# Additional information

- http://www.pgroup.com/lit/articles/insider/v3n1a4.htm
- http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
- https://github.com/parallel-forall/code-samples/blob/master/series/cuda-fortran/async.cuf

WARNING : **cudaFortran is deprecated**

# How to build the cuda/fortran version `async.cuf`

Cuda/Fortran version for Nvidia A100 GPU with architecture sm_80:

```bash
module use /opt/nvidia/hpc_sdk_22_11/modulefiles
module load nvhpc

# compile application
nvfortran -gpu=cuda11.8 -gpu=cc80 -Mcuda=ptxinfo -o async_cuf async.cuf

# run application and collect nsys trace
nsys profile --trace=cuda,nvtx -o nsys_async_cuf --force-overwrite=true ./async_cuf

# visualize nsys trace
nsys-ui nsys_async_cuf.nsys-rep
```

# How to build the Cuda/C++ version `async.cu`

```bash
module use /opt/nvidia/hpc_sdk_22_11/modulefiles
module load nvhpc

# compile application
nvcc -gencode arch=compute_80,code=sm_80 --ptxas-options -v -o async_cu async.cu

# run application and collect nsys trace
nsys profile --trace=cuda,nvtx -o nsys_async_cu --force-overwrite=true ./async_cu

# visualize nsys trace
nsys-ui nsys_async_cu.nsys-rep
```
