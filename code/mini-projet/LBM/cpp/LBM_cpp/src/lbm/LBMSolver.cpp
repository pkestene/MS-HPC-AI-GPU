#include <cstdlib> // for malloc
#include <iostream>
#include <vector>
#include <sstream>

#include "LBMSolver.h"

#include "lbmFlowUtils.h"

#include "writePNG/lodepng.h"
#include "writeVTK/saveVTK.h"

// ======================================================
// ======================================================
LBMSolver::LBMSolver(const LBMParams& params) :
  params(params)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // memory allocations

  // distribution functions
  fin  = (real_t*) malloc(nx*ny*npop * sizeof(real_t));
  fout = (real_t*) malloc(nx*ny*npop * sizeof(real_t));
  feq  = (real_t*) malloc(nx*ny*npop * sizeof(real_t));

  // macroscopic variables
  rho = (real_t*) malloc(nx*ny * sizeof(real_t));
  ux  = (real_t*) malloc(nx*ny * sizeof(real_t));
  uy  = (real_t*) malloc(nx*ny * sizeof(real_t));

  // obstacle
  obstacle = (int *) malloc(nx*ny * sizeof(int));

} // LBMSolver::LBMSolver

// ======================================================
// ======================================================
LBMSolver::~LBMSolver()
{
  // free memory

  // distribution functions
  delete[] fin;
  delete[] fout;
  delete[] feq;

  // macroscopic variables
  delete[] rho;
  delete[] ux;
  delete[] uy;

  // obstacle
  delete[] obstacle;

} // LBMSolver::~LBMSolver

// ======================================================
// ======================================================
void LBMSolver::initialize()
{

  // initialize obstacle mask array
  init_obstacle_mask(params, obstacle);

  // initialize macroscopic velocity
  initialize_macroscopic_variables(params, rho, ux, uy);

  // Initialization of the populations at equilibrium 
  // with the given macroscopic variables.
  equilibrium(params, v, t, rho, ux, uy, fin);
  
} // LBMSolver::initialize

// ======================================================
// ======================================================
void LBMSolver::run()
{

  initialize();

  // time loop
  for (int iTime=0; iTime<params.maxIter; ++iTime) {

    if (iTime % 100 == 0) {
      //output_png(iTime);
      output_vtk(iTime);
    }

    // Right wall: outflow condition.
    // we only need here to specify distrib. function for velocities
    // that enter the domain (other that go out, are set by the streaming step)
    border_outflow(params, fin);
      
    // Compute macroscopic variables, density and velocity.
    macroscopic(params, v, fin, rho, ux, uy);
      
    // Left wall: inflow condition.
    border_inflow(params, fin, rho, ux, uy);

    // Compute equilibrium.
    equilibrium(params, v, t, rho, ux, uy, feq);
    update_fin_inflow(params, feq, fin);

    // Collision step.
    compute_collision(params, fin, feq, fout);

    // Bounce-back condition for obstacle.
    // in python language, we "slice" fout by obstacle
    update_obstacle(params, fin, obstacle, fout);

    // Streaming step.
    streaming(params, v, fout, fin);

  } // end for iTime

} // LBMSolver::run

// ======================================================
// ======================================================
void LBMSolver::output_png(int iTime)
{

  std::cout << "Output data (PNG) at time " << iTime << "\n";

  const int nx = params.nx;
  const int ny = params.ny;

  real_t* u2 = (real_t *) malloc(nx*ny*sizeof(real_t));

  // compute velocity norm, as well as min and max values
  real_t min_value = sqrt(ux[0]*ux[0] + uy[0]*uy[0]);
  real_t max_value = min_value;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      u2[index] = sqrt(ux[index]*ux[index] + uy[index]*uy[index]);

      if (u2[index]<min_value)
        min_value = u2[index];

      if (u2[index]>max_value)
        max_value = u2[index];

    } // end for i

  } // end for j

  // create png image buff
  std::vector<unsigned char> image;
  image.resize(nx * ny * 4);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      // rescale velocity in 0-255 range
      unsigned char value = static_cast<unsigned char>((u2[index]-min_value)/(max_value-min_value)*255);
      image[0 + 4*i + 4*nx*j] = value; 
      image[1 + 4*i + 4*nx*j] = value; 
      image[2 + 4*i + 4*nx*j] = value; 
      image[3 + 4*i + 4*nx*j] = value; 
    }
  }

  std::ostringstream iTimeNum;
  iTimeNum.width(7);
  iTimeNum.fill('0');
  iTimeNum << iTime;

  std::string filename  = "vel_" + iTimeNum.str() + ".png";

  // encode the image
  unsigned error = lodepng::encode(filename, image, nx, ny);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;


  delete[] u2;

} // LBMSolver::output_png

// ======================================================
// ======================================================
void LBMSolver::output_vtk(int iTime)
{

  std::cout << "Output data (VTK) at time " << iTime << "\n";

  bool useAscii = false; // binary data leads to smaller files
  saveVTK(rho, ux, uy, params, useAscii, iTime);

} // LBMSolver::output_vtk
