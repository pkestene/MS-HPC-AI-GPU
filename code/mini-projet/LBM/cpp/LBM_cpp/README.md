# LBM mini project in C++ (CPU)

A minimal cmake based project skeleton for 2D LBM (Lattice Boltzmann Method) simulation.

This C++ code has been adapted from the python code [lbmFlowAroundCylinder.py](https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes/blob/master/lbmFlowAroundCylinder.py)

## about cmake

This template has been derived from [cuda-proj-tmpl](https://github.com/pkestene/cuda-proj-tmpl) so that it is ready for CUDA.

## How to build the serial version ?

```bash
mkdir build
cd build
cmake ..
make
# then you can run the application
./src/lbm
```

## Your task

Derive a parallel version of this code using OpenACC : 

- run and analyze the code, identify the computing kernels containing loops to parallelize with OpenACC directives
- the cmake build is already able to build source code containing OpenACC directives, 
  all you have to do is to uncomment lines annotate with `TODO`
- implement parallelization by provide OpenACC directives in selected location
- re-build the code with Nvidia nvc++ compiler (OpenACC compiler):

```bash
mkdir build_openacc
cd build_openacc
export CXX=nvc++
cmake -DOpenACC_TARGET_FLAGS:STRING="-ta=tesla:cc75 -Minfo=all" ..
make
```

Variable `OpenACC_TARGET_FLAGS` allows you to pass compiler flags to `nvc++` compiler.
