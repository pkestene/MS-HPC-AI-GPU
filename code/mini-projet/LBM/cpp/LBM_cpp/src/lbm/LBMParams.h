#ifndef LBM_PARAMS_H
#define LBM_PARAMS_H

#include "utils/config/ConfigMap.h"
#include "real_type.h"

/**
 * LBM Parameters (declaration)
 */
struct LBMParams {

  //! dimension : 2 or 3
  static const int dim = 2;

  //! number of populations (of distribution functions)
  static const int npop = 9;

  //! run parameters
  int maxIter;

  //! geometry : number of nodes along X axis
  int nx;

  //! geometry : number of nodes along Y axis
  int ny;

  //! physical domain sizes (in lattice units) along X axis
  double lx;

  //! physical domain sizes (in lattice units) along Y axis
  double ly;

  // cylinder obstacle (center coordinates, radius)

  //! x coordinates of cylinder center
  double cx;

  //! y coordinates of cylinder center
  double cy;

  //! cylinder radius
  double r;

  //! initial velocity
  double uLB;

  /*
   * fluid parameters
   */
  //! Reynolds number
  double Re;
  
  //! viscosity in lattice units
  double nuLB;
  
  //! relaxation parameter
  double omega;

  //! setup / initialization
  void setup(const ConfigMap& configMap); 

  //! print parameters on screen
  void print();

}; // struct LBMParams

#endif // LBM_PARAMS_H
