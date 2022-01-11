# LBM mini project in C++ (CPU)

A minimal cmake based project skeleton for 2D LBM (Lattice Boltzmann Method) simulation.

This C++ code has been adapted from the python code [lbmFlowAroundCylinder.py](https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes/blob/master/lbmFlowAroundCylinder.py)

You need to fill TODO's (in directory src/lbm)

# About cmake

This template has been derived from [cuda-proj-tmpl](https://github.com/pkestene/cuda-proj-tmpl) so that it is ready for CUDA.

## Requirements

- cmake version >= 3.18 (when using nvcc)
- cmake version >= 3.22 (when using nvc++/nvhpc compiler)
- cuda toolkit


# How to build ?

## Build with nvcc (Nvidia compiler from CUDA toolkit)


```bash
#  minimum cmake version required is 3.18
module load cmake/3.22.0

# if you want to build with nvcc
module load cuda/11.5
```

```bash
mkdir -p build/nvcc
cd build/nvcc
cmake -DCMAKE_CUDA_ARCHITECTURES="80" ../..
make
# then you can run the application
./src/lbm
```


## Build with nvc++ (Nvidia compiler from hpcsdk package)

```bash
#  minimum cmake version required is 3.22
module load cmake/3.22.0

# if you want to build with nvc++ (from Nvidia hpcsdk)
module load nvhpc/21.11
```

```bash
mkdir -p build/nvcc
cd build/nvcc
export CXX=nvc++
cmake -DCMAKE_CUDA_HOST_COMPILER=nvc++ -DCMAKE_CUDA_ARCHITECTURES="80" ../..
make
# then you can run the application
./src/lbm
```

# Misc

Convert VTK files into python/numpy

```bash
# create a conda env specific to install vtk tools, e.g.
conda create --name vtk --clone numba2021
conda activate vtk
conda install -c conda-forge vtk
```


Then in python, e.g. to convert file `lbm_data_0001900.vti`

```python
import vtk
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("./lbm_data_0001900.vti")
reader.Update()
vtk_data = reader.GetOutput()

# 0 => density
# 1 => vx
# 2 => vy
# 3 => velocity norm

# get velocity norm
vdata = vtk_data.GetPointData().GetArray(3)

# check data sizes
vtk_data.GetExtent()

# convert to numpy array
# take care sizes are swapped
from vtk.util.numpy_support import vtk_to_numpy
v=vtk_to_numpy(vdata).reshape(180,420)

import matplotlib.pyplot as plt
plt.imshow(v)
plt.show()
```
