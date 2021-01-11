# COMPILE

- edit CMakeLists.txt and modify to your need. E.g.
  * enable/disable some executable (comment/uncomment lines starting by cuda_add_executable)
  * enable double precision

- configure and build:
 ```shell
 mkdir build; cd build
 cmake .. (or ccmake ..)
 make
 ```

- type make, this will build the following exe:
* heat_solver_cpu
* heat2d_solver_gpu_naive
* heat2d_solver_gpu_shmem1
* heat2d_solver_gpu_shmem2
* heat3d_solver_gpu_naive
* heat3d_solver_gpu_shmem1
* heat3d_solver_gpu_shmem2

# RUN

For the CPU version:

``` shell
./heat_solver_cpu
```

For the 2D GPU naive version:

``` shell
./heat2d_solver_gpu_naive
```

You can change paramaters by editing heatEqSolver.par


# CUDA DOCUMENTATION

- CUDA Toolkit documentation:
http://docs.nvidia.com/cuda/index.html

- local copy:
/usr/local/cuda/doc/html/index.html


# GOING FURTHER

- use an implicit scheme (e.g. Crank-Nicholson); see tri-diagonal solver on GPU : www.jcohen.name/papers/Zhang_Fast_2009.pdf
- use [cudpp](https://github.com/cudpp/cudpp) and a tridiagonal solver
