# Context

[wave2d](https://github.com/pvthinker/wave2d) is a python code to simulate surface waves (e.g. ship wake) using a FFT-based spectral method.

The computationnal kernels are located in [fourier.py](https://github.com/pvthinker/wave2d/blob/master/fourier.py)

# Work

- Refactoring this code by using as much as possible [CuPy](https://github.com/cupy/cupy) instead of [numpy](https://numpy.org/), and try to evaluate performance (change simulation domain sizes).
- By how much the CuPy version is faster than the original CPU only version ? More precisely perform a benchmark by varying the domain size, perform a strong scaling study. Beware to deactivate animation when performing performance measurements.
- CuPy.fft is able to handle multi-GPU computation, investigate the use of the two GPUs available on our platform.

Main files are [wave2d.py](https://github.com/pvthinker/wave2d/blob/master/wave2d.py), [fourier.py](https://github.com/pvthinker/wave2d/blob/master/fourier.py) and [shipwake.py](https://github.com/pvthinker/wave2d/blob/master/shipwake.py)

Additional resources:
- [CuPy](https://github.com/cupy/cupy)
- [GTC 2020 Numba and CuPy tutorial](https://github.com/ContinuumIO/gtc2020-numba)
- [Fast Fourier Transform with CuPy](https://docs.cupy.dev/en/stable/user_guide/fft.html)
- [ondes dans les fluides geophysiques](http://stockage.univ-brest.fr/~roullet/documents/cours_ondes_2012.pdf)
- [notion de relation de dispersion sur les ondes dans les fluides](https://en.wikipedia.org/wiki/Dispersion_(water_waves)) : this explains (without to much details) the (often non-linear) relation between the pulsation `omega` and the wave vector `k` (at the heart of the Fourier-based simulation method used here).
- [wave2d in Matlab](http://stockage.univ-brest.fr/~roullet/documents/wave2D.m), the same numerical method coded in Matlab (shorter to read, might help to understand the numerical method).
