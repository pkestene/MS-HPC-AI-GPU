/**
 * \file heat2d_solver_gpu_naive_managed.cu
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

#include <sys/time.h> // gettimeofday

// includes, project
//#include <helper_functions.h>
#include "cuda_error.h"
#include "CudaTimer.h"
#include "Timer.h"

// parameters + real_t typedef
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

  cudaDeviceSynchronize();
  cudaDeviceReset();

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
  unsigned int mem_size = sizeof(real_t)*NX*NY;

  CudaTimer gpuTimer;

  // store GPU results for comparison with CPU
  real_t *resGPU = (real_t*) malloc(mem_size);

  int iTime=0;

  ///////////////////////////////////////////////////
  // compute GPU solution to 2D heat equation
  ///////////////////////////////////////////////////
  {
    // allocate memory for both Host and Device !
    real_t* data1; cudaMallocManaged(&data1, mem_size);
    real_t* data2; cudaMallocManaged(&data2, mem_size);
    
    
    // inital condition
    initCondition2D (data1);
    initCondition2D (data2);
    
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
    int iOutput=-1;
    for (real_t t=0.0f; t<TMAX; t+=(2*DT), iTime+=2) {
    
      if (useOrder2) { // use the 2nd order accurate scheme
      
	heat2d_ftcs_naive_order2_kernel<<< grid, threads >>>( data1, data2, 
							      NX, NY,
							      o2.R, o2.R2);
        CUDA_KERNEL_CHECK("Kernel execution failed");
      
	heat2d_ftcs_naive_order2_kernel<<< grid, threads >>>( data2, data1, 
							      NX, NY,
							      o2.R, o2.R2);   
        CUDA_KERNEL_CHECK("Kernel execution failed");
      
      } else if (useOrder2b) { // use the 2nd order accurate scheme
      
	heat2d_ftcs_naive_order2b_kernel<<< grid, threads >>>( data1, data2, 
							       NX, NY,
							       o2.R, o2.R2b);
        CUDA_KERNEL_CHECK("Kernel execution failed");
      
	heat2d_ftcs_naive_order2b_kernel<<< grid, threads >>>( data2, data1, 
							       NX, NY,
							       o2.R, o2.R2b);   
        CUDA_KERNEL_CHECK("Kernel execution failed");

      } else { // use the 4th order accurate scheme
      
	heat2d_ftcs_naive_order4_kernel<<< grid, threads >>>( data1, data2, 
							      NX, NY,
							      o4.S, o4.S2);
        CUDA_KERNEL_CHECK("Kernel execution failed");
      
	heat2d_ftcs_naive_order4_kernel<<< grid, threads >>>( data2, data1, 
							      NX, NY,
							      o4.S, o4.S2);   
	CUDA_KERNEL_CHECK("Kernel execution failed");
      }

      /* save output (just for cross-checking, do not save when
	 measuring computing time */
      if (ENABLE_GPU_SAVE) {

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

    // copy result from device to host
    CUDA_API_CHECK( cudaMemcpy( resGPU, data1, NX*NY*sizeof( real_t),
                                cudaMemcpyDeviceToHost) );

    // cleanup memory
    cudaFree(data1);
    cudaFree(data2);

  }
  // ending GPU computation


  real_t gpu_time = gpuTimer.elapsed();
  printf( "GPU Processing time: %f (s)\n", gpu_time);
      
  if (SAVE_HDF5)
    write_xdmf_wrapper("heat2d_gpu_naive",N_ITER,T_OUTPUT);
 

  Timer cpuTimer;
  // store CPU results for comparison with GPU
  real_t *resCPU = (real_t*) malloc(mem_size);

  ////////////////////////////////////////////////////////
  // compute reference (CPU) solution to 2D heat equation
  // for performance comparison
  ////////////////////////////////////////////////////////
  {

    real_t *data1 = (real_t *) malloc(mem_size);
    real_t *data2 = (real_t *) malloc(mem_size);

    initCondition2D (data1);
    initCondition2D (data2);
    
    cpuTimer.start();
    
    // time loop
    iTime=0;

    for (real_t t=0.0f; t<TMAX; t+=(2*DT), iTime+=2) {
    
      // compute next 2 time steps
      if (useOrder2) {
	heat2d_ftcs_cpu_order2( data1, data2);
	heat2d_ftcs_cpu_order2( data2, data1);
      } else if (useOrder2b) {
	heat2d_ftcs_cpu_order2b( data1, data2);
	heat2d_ftcs_cpu_order2b( data2, data1);
      } else {
	heat2d_ftcs_cpu_order4( data1, data2);
	heat2d_ftcs_cpu_order4( data2, data1);
      }
    }

    // stop timer
    cpuTimer.stop();

    // copy CPU results
    memcpy(resCPU, data1, mem_size);

    // cleanup memory
    free(data1);
    free(data2);

  }

  real_t cpu_time = cpuTimer.elapsed();
  
  printf( "CPU Processing time: %g (s)\n", cpu_time);
  printf( "Speedup GPU/CPU : %f\n",cpu_time/gpu_time);

  printf("...comparing the results\n");
  double sum = 0, delta = 0;
  for(unsigned i = 0; i < NX*NY; i++){
    delta += (resGPU[i] - resCPU[i]) * (resGPU[i] - resCPU[i]);
    sum   +=  resCPU[i] * resCPU[i];
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
  free(resCPU);
  free(resGPU);
    
  return EXIT_SUCCESS;
}
