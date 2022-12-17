# Mini-projects about using Nvidia's Modulus framework

For all the mini-projects:
- provide a short report (~ 2 to 4 pages)
- provide a URL to a git repository used to developed your project with a short readme to use your code

## Allen-Cahn

Allen-Cahn equations are partial differential equations of type reaction-diffusion and are used for example to model spontaneous phase separation phenomena.
These equations are stiff, i.e. hard to solve numerically.

Your task:

- read article https://arxiv.org/pdf/2007.04542.pdf
- using Modulus re-implement a solver for Allen-Cahn PDE (using the same setup for initial and border conditions)
- have a look at Modulus documentation and examples to get familiar with the monitoring subpackage which you can use to extract solutions and be able to reproduce plots like the ones in figures 3.1 to 3.3 (from the article above)
- extract the loss function evolution during training to cross-check training is actually convergent
- can you assess if the solution ouput by the trained  neural network is correct ? If not, have a look at the following page in Modulus documentation (Best Practices) : the [Signed Distance Function](https://docs.nvidia.com/deeplearning/modulus/user_guide/theory/recommended_practices.html) can be use to help training to converge (see also examples inside dedicated modulus examples repository)
- you can also perform a quantitative study of the quality and performance of the training when changing parameters like : the number of hidden layers, the number of nodes per hidden layer.
- you may also compare your Modulus-based code with https://github.com/maziarraissi/PINNs/blob/master/main/discrete_time_inference%20(AC)/AC.py
- you may also try to implement a gPINN (gradient-enhanced PINN) variant, cf : https://arxiv.org/pdf/2111.02801.pdf by introducing partial derivatives of the PDE residu into the loss function.

## Eikonal

eikonal equation is a PDE which can be found for example in geometric optics, in acoustics, etc...
In optics, this equation fundamentaly determine the light ray trajectories in a media where refractive index can be a non-uniform function of space.

Your task:

- read the following article:  https://arxiv.org/abs/2007.08330
- try to reproduce its results using the same numerical setup (initial and border conditions)
- perform a quantitative study of the solution quality, a study of the influence of the neural network parameters: number of hidden layers, size of hidden layer, learning rate, batch size etc....
- did-you need to constraint the colocation point sampling used during training (i.e. parameter lambda_weighting from class PointwiseInteriorConstraint) to enforce training convergence ?
- provide a short report (~ 2 to 4 pages) with all you findings, with some inference results (solution predicted by the neural networks) to illustrate that your implementation is compatible with results from the article;
- you may read Modulus documentation to learn how to use class PointwiseInferencer which will allow you to make graphics outputs (VTK format) which you can use to make visualization with Paraview.

## PINO applied to 2D wave equations

PINO = Physics Informed Neural Operator
Given a partial differential equations system, PINO allow among to approximate a operator which maps the initial conditions to the final solution of the PDE.

- follow PINO tutorial using modulus on Darcy equations (see github.com/ openhackathons-org /
gpubootcamp) : https://github.com/openhackathons-org/gpubootcamp/blob/master/hpc_ai/PINN/English/python/jupyter_notebook/Operators/Darcy_Flow_using_Physics_Informed_Neural_Operator.ipynb
- read the following article https://arxiv.org/pdf/2203.12634.pdf (Applications of physics informed neural operators)

Your task :
- using modulus, implement a PINO solver for the 2D wave equation by adapting from Darcy
- provide a report detailing what you have learned about PINO, your setup, a qualitative study of the influence of the training parameters upon the solution
