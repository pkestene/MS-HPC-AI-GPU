# Purpose

Explore the benefit of using library [MatX](https://github.com/NVIDIA/MatX) for developing a new GPU-accelerated scientific application (or simply refactoring an existing one) and compare with [CuPy](https://docs.cupy.dev/en/stable/index.html).

Some resources:
- [MatX documentation](https://nvidia.github.io/MatX/)
- [CuPy documentation](https://docs.cupy.dev/en/stable/)

# Activities

## Preliminary

Learn how to use MatX

- clone, build and install MatX; build with examples and benchmarch
- get familiar with MatX, use provided [notebooks](https://github.com/NVIDIA/MatX/tree/main/docs_input/notebooks)
- Create a standalone project by using the provided [template](https://github.com/NVIDIA/MatX/tree/main/examples/cmake_sample_project)
- try to cross-check the fact that MatX can be faster than CuPy by using example `resample` (see MatX readme)

## CuPy / MatX comparison

- take at least one [Cupy example](https://github.com/cupy/cupy/tree/master/examples), e.g. `cg` (conjugate gradient) and refactor it with MatX
- provide a perform study comparison (regular Numpy on CPU; CuPy and MatX)
