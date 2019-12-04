#ifndef LBM_SOLVER_H
#define LBM_SOLVER_H

#include "real_type.h"
#include "LBMParams.h"
#include "lbmFlowUtils.h"

/**
 * class LBMSolver for D2Q9
 *
 * Adapted and translated to C++ from original python version
 * found here :
 * https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes
 *
 * LBM lattice : D2Q9
 *
 * 6   3   0
 *  \  |  /
 *   \ | /
 * 7---4---1
 *   / | \
 *  /  |  \
 * 8   5   2
 *
 */
class LBMSolver
{

public:
  LBMSolver(const LBMParams& params);
  ~LBMSolver();

  //! LBM weight for D2Q9
  const weights_t t{1.0/36, 1.0/9, 1.0/36, 1.0/9, 4.0/9, 1.0/9, 1.0/36, 1.0/9, 1.0/36};

  // LBM lattive velocity (X and Y components) for D2Q9
  const velocity_array_t v{
    1,  1, 
    1,  0,
    1, -1, 
    0,  1,
    0,  0, 
    0, -1, 
    -1,  1,
    -1,  0, 
    -1, -1};

  // distribution functions
  real_t* fin;
  real_t* fout;
  real_t* feq;

  // macroscopic variables
  real_t* rho;
  real_t* ux;
  real_t* uy;

  // obstacle
  int* obstacle;

  const LBMParams& params;

  void initialize();
  void run();
  void output_png(int iTime);
  void output_vtk(int iTime);

}; // class LBMSolver

#endif // LBM_SOLVER_H
