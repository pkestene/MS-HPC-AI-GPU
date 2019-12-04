/**
 * \file heat2d_solver_gpu_naive.cu
 * \brief Solve 2D heat equation (finite difference method). GPU version (naive).
 *
 * We solve the 2D Heat equation \f$\partial_t \phi = \alpha \left[
 * \partial^2_x \phi + \partial^2_y \phi \right] \f$, \f$ 0 \leq x
 * \leq L_x \f$, \f$ 0 \leq y \leq L_y \f$, \f$ 0 \leq t\f$.\\
 *
 * Method : Finite Difference, FTCS scheme
 *
 * GPU Features: use only global memory
 *
 * boundary condition : Dirichlet
 *
 * GPU version : naive
 *
 * \date 17-dec-2009.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h> // for gettimeofday

#ifdef _OPENMP
#include <omp.h>
#endif
#include "openmp_utils.h"

// includes, project
//#include <helper_functions.h>
#include "cuda_error.h"
#include "CudaTimer.h"
#include "Timer.h"

// parameters and real_t typedef
#include "param.h"

// for output results
#include "output.h"

// GPU solver
#include "heat2d_kernel_gpu_naive.cu"

// CPU solver
#include "heat_kernel_cpu.h"

// initial conditions
#include "misc.h"

// cuda helper
#include "cuda_helper.h"

/////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest( int argc, char** argv);

/////////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{
  int status = runTest(argc, argv);

  CUDA_API_CHECK ( cudaDeviceSynchronize() );
  CUDA_API_CHECK ( cudaDeviceReset() );

  return status;
}

/////////////////////////////////////////////////////////////////////////
//! Run solver on GPU
/////////////////////////////////////////////////////////////////////////
int
runTest(int argc, char** argv) 
{
  int devID;
  cudaDeviceProp deviceProps;
  
  devID = findCudaDevice(argc, (const char **)argv);
  
  // get number of SMs on this GPU
  CUDA_API_CHECK( cudaGetDeviceProperties(&deviceProps, devID) );
  printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

  /*
   * read and print parameters
   */
  // default parameter file
  std::string paramFile("heatEqSolver.par");

  // if argv[1] exists use it as a parameter file
  if (argc>1) {
    printf("trying to read parameters from file %s ...\n",argv[1]);
    paramFile = std::string(argv[1]);
  }

  // read parameter file
  readParamFile(paramFile);

  // print parameters on screen
  printParameters("HEAT 2D - GPU (NAIVE)");

  CudaTimer gpuTimer;

  unsigned int mem_size = sizeof(real_t)*NX*NY;

  // allocate host memory
  real_t* data1 = (real_t*) malloc( mem_size);
  real_t* data2 = (real_t*) malloc( mem_size);
  
  ///////////////////////////////////////////////////
  // compute GPU solution to 2D heat equation
  ///////////////////////////////////////////////////
  
  // inital condition
  initCondition2D (data1);

  // allocate device memory
  real_t* d_data1;
  real_t* d_data2;

  // device memory allocation (using cudaMalloc)
  CUDA_API_CHECK( cudaMalloc( (void**) &d_data1, mem_size));
  CUDA_API_CHECK( cudaMalloc( (void**) &d_data2, mem_size));

  // copy host memory to device
  CUDA_API_CHECK( cudaMemcpy( d_data1, data1, mem_size,
                              cudaMemcpyHostToDevice) );
  CUDA_API_CHECK( cudaMemcpy( d_data2, data1, mem_size,
                              cudaMemcpyHostToDevice) );
   
  // setup execution parameters for cuda kernel
  // grid dimension for naive kernel
  unsigned int threadsPerBlockX=32;
  unsigned int threadsPerBlockY=10;
  // unsigned int threadsPerBlockX=64;
  // unsigned int threadsPerBlockY=4;
  dim3  threads(threadsPerBlockX,threadsPerBlockY);
  dim3  grid( (NX+threads.x-1)/threads.x, (NY+threads.y-1)/threads.y );
    
  printf("grid  size : %u %u\n",grid.x,grid.y);
  printf("block size : %u %u\n",threads.x,threads.y);

  // start timer
  gpuTimer.start();

  // time loop executing naive kernel
  int iTime   =  0;
  int iOutput = -1;
  for (iTime=0; iTime*DT<TMAX; ++iTime) {

    real_t* data_in  = iTime%2 == 0 ? d_data1 : d_data2;
    real_t* data_out = iTime%2 == 0 ? d_data2 : d_data1;

    if (useOrder2) { // use the 2nd order accurate scheme
      
      heat2d_ftcs_naive_order2_kernel<<< grid, threads >>>( data_in, 
                                                            data_out,
							    NX, NY,
							    o2.R, o2.R2);
      CUDA_KERNEL_CHECK("Kernel execution failed");
      
    } else if (useOrder2b) { // use the 2nd order accurate scheme
      
      heat2d_ftcs_naive_order2b_kernel<<< grid, threads >>>( data_in, 
                                                             data_out,
							     NX, NY,
							     o2.R, o2.R2b);
      CUDA_KERNEL_CHECK("Kernel execution failed");
      
    } else { // use the 4th order accurate scheme
      
      heat2d_ftcs_naive_order4_kernel<<< grid, threads >>>( data_in,
                                                            data_out,
							    NX, NY,
							    o4.S, o4.S2);
      CUDA_KERNEL_CHECK("Kernel execution failed");
      
    }

    /* save output (just for cross-checking, do not save when
       measuring computing time */
    if (ENABLE_GPU_SAVE) {

      if (iTime%T_OUTPUT == 0) {
	iOutput++;
	CUDA_API_CHECK( cudaMemcpy( data1, data_out, NX*NY*sizeof( real_t),
                                    cudaMemcpyDeviceToHost) );      
      }
      // PGM output
      if (SAVE_PGM and iTime%T_OUTPUT == 0)
	save_pgm(data1, "heat2d_gpu_naive_",iOutput,NX,NY);
      
      // MathGL save (3D view)
      if (SAVE_MGL and iTime%T_OUTPUT == 0)
	save_mgl(data1, "heat2d_gpu_naive_",iOutput,NX,NY);

      // VTK output
      if (SAVE_VTK and iTime%T_OUTPUT == 0)
	save_vtk(data1, "heat2d_gpu_naive_",iOutput);

      // HDF5 output
      if (SAVE_HDF5 and iTime%T_OUTPUT == 0)
	save_hdf5(data1, "heat2d_gpu_naive_",iOutput);

    }

  } // end for loop
  
  // stop timer
  gpuTimer.stop();

  real_t gpu_time = gpuTimer.elapsed();
  printf( "GPU Processing time: %f (s)\n", gpu_time);
  
  // copy result from device to host
  real_t *resGPU = (real_t*) malloc( mem_size);
  CUDA_API_CHECK( cudaMemcpy( resGPU, d_data1, NX*NY*sizeof( real_t),
                              cudaMemcpyDeviceToHost) );
    
  if (SAVE_HDF5)
    write_xdmf_wrapper("heat2d_gpu_naive",N_ITER,T_OUTPUT);
 
  ////////////////////////////////////////////////////////
  // compute reference (CPU) solution to 2D heat equation
  // for performance comparison
  ////////////////////////////////////////////////////////
  printf("compute CPU reference solution\n");
  initCondition2D (data1);
  initCondition2D (data2);

  print_openmp_status();

  Timer cpuTimer;
  cpuTimer.start();
  
  // time loop
  iTime=0;
  for (iTime=0; iTime*DT<TMAX; ++iTime) {
    
    real_t* data_in  = iTime%2 == 0 ? data1 : data2;
    real_t* data_out = iTime%2 == 0 ? data2 : data1;

    if (useOrder2) {
    
      heat2d_ftcs_cpu_order2( data_in, data_out );
      
    } else if (useOrder2b) {

      heat2d_ftcs_cpu_order2b( data_in, data_out );
      
    } else {

      heat2d_ftcs_cpu_order4( data_in, data_out );
      
    }
  }

  // stop timer
  cpuTimer.stop();
  real_t cpu_time = cpuTimer.elapsed();
  
  printf( "CPU Processing time: %g (s)\n", cpu_time);
  printf( "Speedup GPU/CPU : %f\n",cpu_time/gpu_time);

  printf("...comparing the results\n");
  double sum = 0, delta = 0;
  for(unsigned i = 0; i < NX*NY; i++){
    delta += (resGPU[i] - data1[i]) * (resGPU[i] - data1[i]);
    sum   += data1[i] * data1[i];
  }
  double L2norm = sqrt(delta / sum);
  printf("iteration %d relative L2 norm: %E\n", iTime, L2norm);

  // cuda device prop (to compute max bandwidth
  cudaDeviceProp deviceProp;
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&deviceProp, deviceId);
  //printf("GPU mem clock rate in kHz %d\n",deviceProp.memoryClockRate);
  //printf("GPU mem bus width %d\n",deviceProp.memoryBusWidth);
  real_t gpuMaxBW = 1e-9*deviceProp.memoryClockRate*1000*(deviceProp.memoryBusWidth/8)*2;
  
  // bandwidth
  double totalBytes;
  if (useOrder2)
    totalBytes = NX*NY*sizeof(real_t) * (5 + 1) * iTime;
  else if (useOrder2b)
    totalBytes = NX*NY*sizeof(real_t) * (3*3 + 1) * iTime;
  else
    totalBytes = NX*NY*sizeof(real_t) * (9 + 1) * iTime;
  printf("CPU Bandwidth %f GBytes/s\n", totalBytes/cpu_time*1e-9);
  printf("GPU Bandwidth %f GBytes/s out of %f (%6.2f %%)\n", totalBytes/gpu_time*1e-9,gpuMaxBW,totalBytes/gpu_time*1e-9/gpuMaxBW*100);

  // cleanup memory
  free(data1);
  free(data2);
  free(resGPU);
  
  CUDA_API_CHECK( cudaFree(d_data1) );
  CUDA_API_CHECK( cudaFree(d_data2) );
  
  return EXIT_SUCCESS;
}
