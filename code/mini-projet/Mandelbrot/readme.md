Revisit / refactor code located here:

https://github.com/pkestene/AMR_mandelbrot

Change the data structure for using one of the following alternative hash table implementation for GPU:
- https://github.com/NVIDIA/cuCollections; try to use `cuco::dynamic_map` data structure
- https://github.com/owensgroup/BGHT and read article https://arxiv.org/pdf/2108.07232.pdf
- https://github.com/sleeepyjack/warpcore and read https://arxiv.org/pdf/2009.07914.pdf

Use a performance comparison study to investigate if using one of the two hash table implementations alternative is interesting compared to the `Kokkos::Unordered_Map` data structure
