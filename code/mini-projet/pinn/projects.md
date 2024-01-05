# Mini-projects about PINNs

For all the mini-projects:
- provide a video 12 to 15 minutes max reporting your work
- provide a URL to a git repository used to developed your project with a short readme to use your code
- deadline: 2024, 15 March.

# PDEBench / Nvidia Modulus

- read article  https://arxiv.org/pdf/2210.07182.pdf which describe [PDEBench](https://github.com/pdebench/PDEBench)
- install PDEBench on hpcai using a dedicated conda envrironment; you'll  probably have to install also pytorch and/or JAX in the same environment
- chose one the applications used in PDEBench:
  - 1D advection equation
  - 1D burgers equation
  - 1D reaction diffusion
  - 2D shallow water
- use PDEBench for data generation, training, inference, and performance mesurements
- re-implement the same model using Nvidia/Modulus
- compare Modulus and PDEBench (performances, results quality, ...) using the metrics described in the article.
