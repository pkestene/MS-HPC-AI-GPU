/**
 * \file heat_solver_cpu.cpp
 * \brief Solve heat equation (finite difference method). CPU version.
 *
 * We solve the 2D / 3D Heat equation \f$\partial_t \phi = \alpha \left[
 * \partial^2_x \phi + \partial^2_y \phi \right] \f$, \f$ 0 \leq x
 * \leq L_x \f$, \f$ 0 \leq y \leq L_y \f$, \f$ 0 \leq t\f$.\\
 *
 * Method : Finite Difference, FTCS scheme
 *
 * boundary condition : Dirichlet
 *
 * \date 17-dec-2009.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, application
#include "param.h"
#include "output.h"
#include "misc.h"
#include "heat_kernel_cpu.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <openmp_utils.h>

int
main(int argc, char** argv) 
{

  // default parameter file
  std::string paramFile("heatEqSolver.par");

  // if argv[1] exists use it as a parameter file
  if (argc>1) {
    printf("trying to read parameters from file %s ...\n",argv[1]);
    paramFile = std::string(argv[1]);
  }

  // read parameter file
  readParamFile(paramFile);

  // print parameters
  printParameters("HEAT solver on CPU");

  print_openmp_status();

  // use NZ=1 to do 2D simulations
  unsigned int mem_size = sizeof(real_t) * NX * NY * NZ;

  // allocate host memory
  real_t* data1 = (real_t*) malloc(mem_size);
  real_t* data2 = (real_t*) malloc(mem_size);

  // initalize the memory
  if (NZ==1) {
    initCondition2D(data1);
    initCondition2D(data2);
  } else { 
    initCondition3D(data1);
    initCondition3D(data2);
  }

  // compute reference solution on CPU
  int iTime=0;
  int iOutput=-1;
  
  if (NZ==1) {
    for (int iTime=0; 
         iTime*DT<TMAX; 
         ++iTime) {

      real_t* data_in  = iTime%2 == 0 ? data1 : data2;
      real_t* data_out = iTime%2 == 0 ? data2 : data1;

      if (iTime%T_OUTPUT==0) {
	printf("time : %d (%16.14f)\n",iTime,iTime*DT);
	iOutput++;
      }

      // compute time step
      if (useOrder2) {
	heat2d_ftcs_cpu_order2(data_in, data_out);
      } else if (useOrder2b) {
	heat2d_ftcs_cpu_order2b(data_in, data_out);
      } else {
	heat2d_ftcs_cpu_order4(data_in, data_out);
      }
      
      // PGM output
      if (SAVE_PGM and iTime%T_OUTPUT == 0)
	save_pgm(data_out, "heat2d_cpu_",iOutput,NX,NY);
      
      // MathGL save (3D view)
      if (SAVE_MGL and iTime%T_OUTPUT == 0)
	save_mgl(data_out, "heat2d_cpu_",iOutput,NX,NY);

      // VTK output
      if (SAVE_VTK and iTime%T_OUTPUT == 0)
	save_vtk(data_out, "heat2d_cpu_",iOutput);

      // HDF5 output
      if (SAVE_HDF5 and iTime%T_OUTPUT == 0)
	save_hdf5(data_out, "heat2d_cpu_",iOutput);

    } // end for time

  } else { // 3D

    for (int iTime=0;
         iTime*DT<TMAX; 
         ++iTime) {
      
      real_t* data_in  = iTime%2 == 0 ? data1 : data2;
      real_t* data_out = iTime%2 == 0 ? data2 : data1;

      if (iTime%T_OUTPUT==0) {
	printf("time : %d (%8f)\n",iTime,iTime*DT);
	iOutput++;
      }
      
      // compute time step
      if (useOrder2) {
	heat3d_ftcs_cpu_order2(data_in, data_out);
      } else if (useOrder2b) {
	heat3d_ftcs_cpu_order2b(data_in, data_out);
      } else {
	heat3d_ftcs_cpu_order4(data_in, data_out);
      }
           
      // MathGL save (3D view)
      if (SAVE_MGL and iTime%T_OUTPUT == 0)
	save_mgl_3d(data_out, "heat3d_cpu_",iOutput,NX,NY,NZ);
      
      // VTK output
      if (SAVE_VTK and iTime%T_OUTPUT == 0)
	save_vtk(data_out, "heat3d_cpu_",iOutput);

      // HDF5 output
      if (SAVE_HDF5 and iTime%T_OUTPUT == 0)
	save_hdf5(data_out, "heat3d_cpu_",iOutput);

   } // end for time
    
  }

  if (SAVE_HDF5) {
    if (NZ==1)
      write_xdmf_wrapper("heat2d_cpu",N_ITER,T_OUTPUT);
    else
      write_xdmf_wrapper("heat3d_cpu",N_ITER,T_OUTPUT);
  }

  // cleanup memory
  free(data1);
  free(data2);

  return EXIT_SUCCESS;
}
