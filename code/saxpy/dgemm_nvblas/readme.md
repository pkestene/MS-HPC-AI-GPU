# NVBLAS documentation

https://docs.nvidia.com/cuda/nvblas/index.html

Here we use the GPU with recompiling application, no code change is required.

# Build application for CPU

## Build with icc+mkl for CPU execution:

```shell
icc -o dgemm_example_icc dgemm_example.c -mkl
```

## Build with gcc+mkl for CPU execution:

see https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

Note that MKL library can be used even without the Intel compiler

```shell
gcc -DMKL_ILP64 -fopenmp -m64 -I$MKL_ROOT/include -o dgemm_example_gnu dgemm_example.c  -Wl,--no-as-needed -L$MKL_ROOT/lib/intel64 -L$INTEL_PROD_DIR/compiler/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread -lm
```

# Run on GPU without recompiling

See doc https://docs.nvidia.com/cuda/nvblas/index.html#Usage

Here we use NVIDIA nvblas as a drop-in replacement for MKL without recompiling and run on GPU. We just need to pre-load shared library libnvblas.so which will intercept all cpu blas call, and execution nvblas instead

```shell
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvblas.so ./dgemm_example_gnu
```

You can check with profiling tool _nsys_ that your GPU is actually used:

1. profile execution

```shell
nsys profile --trace=cuda,cublas -w true --stats=true -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvblas.so ./dgemm_example_gnu
```

2. open profiling data with _nsight-sys_ (see log of step to know the path to output profile file with extension qdrep)

```shell
nsight-sys /home/pkestene/nvidia_nsight_systems/report16.qdrep
```


# Additionnal documentation

See also slide 21 of doc
http://www.csm.ornl.gov/workshops/openshmem2015/documents/GPGPU_tutorial_2015.pdf
