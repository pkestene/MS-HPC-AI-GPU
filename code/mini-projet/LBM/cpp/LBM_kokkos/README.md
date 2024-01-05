# LBM mini project in C++ (CPU)

A minimal cmake based project skeleton for 2D LBM (Lattice Boltzmann Method) simulation.

This C++ code has been adapted from the python code [lbmFlowAroundCylinder.py](https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes/blob/master/lbmFlowAroundCylinder.py)

## How to build the serial version ?

There is a serial version of this code located in `../LBM_cpp`

How to build serial version:
```bash
mkdir build
cd build
cmake ..
make
# then you can run the application
./src/lbm
```

## Your task

Implement a parallel version of this code using kokkos :

- run and analyze the serial code, identify the computing kernels containing loops to parallelize with kokkos
- identify which data array are use inside computing kernels; transform them into `Kokkos::View`
- the cmake build is already able to build kokkos (or use kokkos modulefiles)
- implement parallelization using Kokkos: don't forget to add `Kokkos::initialize` / `Kokkos::finalize` in the main
- in folder src/lbm, replace header `real_type.h` by `real_type_kokkos.h`


Example build for Kokkos::OpenMP backend

```bash
mkdir -p _build/kokkos_openmp
cd _build/kokkos_openmp
cmake -DLBM_KOKKOS_BUILD:BOOL=ON -DLBM_KOKKOS_BACKEND=OpenMP ../..
make
```

Example build for Kokkos::Cuda backend (you need to load modulefile cuda )

```bash
# cuda module on hpcai
module load cuda/11.8

mkdir -p _build/kokkos_cuda
cd _build/kokkos_cuda
cmake -DLBM_KOKKOS_BUILD:BOOL=ON -DLBM_KOKKOS_BACKEND=Cuda ../..
make
```
