#include "LBMParams.h"


// ======================================================
// ======================================================
void LBMParams::setup(const ConfigMap& configMap)
{

  // initialize run parameters
  maxIter = configMap.getInteger("run","maxIter",2000);

  // geometry
  nx = configMap.getInteger("geometry","nx",420);
  ny = configMap.getInteger("geometry","ny",180);

  lx = static_cast<double>(nx)-1;
  ly = static_cast<double>(ny)-1;

  // cylinder
  cx = configMap.getFloat("cylinder","cx",1.0*nx/4);
  cy = configMap.getFloat("cylinder","cy",1.0*ny/2);
  r  = configMap.getFloat("cylinder","r",1.0*ny/9);

  // fluids parameters

  // initial velocity
  uLB = configMap.getFloat("fluid","uLB",0.04);

  // Reynolds number.
  Re = configMap.getFloat("fluid","Re",150.0);

  // Viscoscity in lattice units.
  nuLB = uLB * r / Re;

  // Relaxation parameter.
  omega = 1.0/(3*nuLB+0.5);

} // LBMParams::setup

// ======================================================
// ======================================================
void LBMParams::print()
{

  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  printf( "nx         : %d\n", nx);
  printf( "ny         : %d\n", ny);
  printf( "lx         : %f\n", lx);
  printf( "ly         : %f\n", ly);
  printf( "cx         : %f\n", cx);
  printf( "cy         : %f\n", cy);
  printf( "r          : %f\n", r );
  printf( "uLB        : %f\n", uLB);
  printf( "Re         : %f\n", Re);
  printf( "nuLB       : %f\n", nuLB);
  printf( "omega      : %f\n", omega);
  printf( "\n");

} // LBMParams::print
