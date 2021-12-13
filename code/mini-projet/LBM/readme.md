Mini-project LBM on GPU

# Introduction to Lattice Boltzmann Method

Many references, e.g.
- https://tel.archives-ouvertes.fr/tel-00311293v2/document

Books:
- https://www.springer.com/gp/book/9783319446479
- You can read [chapter 3](https://link.springer.com/content/pdf/10.1007%2F978-3-319-44649-3_3.pdf)

Github:
- code https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes
- slides https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes/blob/master/W_5_Lattice-Boltzmann-modeling-of-fluid-flow.pdf
- [Code](https://github.com/lbm-principles-practice/code) accompagnying the book [The Lattice Boltzmann method: principles and practive](https://link.springer.com/book/10.1007/978-3-319-44649-3)

Video:
- [Palabos MOOC on LBM](https://palabos.unige.ch/lattice-boltzmann/what-lattice-boltzmann/)

Additional references about border conditions in LBM:
- http://etheses.whiterose.ac.uk/13546/1/Thesis_for_web_new.pdf

# Porting LBM simulation to GPU

You have three choices for the mini-project:

- use pure CUDA/C++ to adapt the C++ serial version from directory `cpp`
- use OpenACC to port the C++ serial version
- use kokkos or stdpar (for the brave) to port the C++ version to GPU
- use numba and/or CuPy to adapt the python code `python/lbmFlowAroundCylinder.py`
- use legate/cunumeric to adapt the python code `python/lbmFlowAroundCylinder.py` (dont be afraid, some feature are not yet implemented in legate/cunumeric, but it is possible to slightly refactor python to make it possible)
