/**
 * \file heat3d_solver_gpu_shmem4.cu
 * \brief Solve 3D heat equation (finite difference method). GPU version (shmem4).
 *
 * We solve the 3D Heat equation \f$\partial_t \phi = \alpha \left[
 * \partial^2_x \phi + \partial^2_y \phi + \partial^2_z \ phi \right] \f$, \f$ 0 \leq x
 * \leq L_x \f$, \f$ 0 \leq y \leq L_y \f$, \f$ 0 \leq t\f$.\\
 *
 * Method : Finite Difference, FTCS scheme
 *
 * boundary condition : Dirichlet
 *
 * GPU version : shared memory 4
 *
 * \date 20-nov-2012.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h> // for gettimeofday
#include <assert.h>

// includes, project
#include <helper_functions.h>
#include "CudaTimer.h"
#include "Timer.h"

// parameters and real_t typedef
#include "param.h"

// for output results
#include "output.h"

// GPU solver
#include "heat3d_kernel_gpu_shmem4.cu"

// CPU solver
#include "heat_kernel_cpu.h"

// initial conditions
#include "misc.h"

// cuda helper
#include "cuda_helper.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{
  runTest(argc, argv);

  exit(0);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv) 
{
  int devID;
  cudaDeviceProp deviceProps;
  
  devID = findCudaDevice(argc, (const char **)argv);
  
  // get number of SMs on this GPU
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
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

  if (NZ<=1) {
    printf("NZ should be larger than 1 in the 3D version\n");
    cudaDeviceReset();
  }

  // print parameters on screen
  printParameters("HEAT 3D - GPU (SHMEM4)");

  CudaTimer gpuTimer;

  unsigned int mem_size = sizeof(real_t)*NX*NY*NZ;

  // allocate host memory
  real_t* data1 = (real_t*) malloc( mem_size);
  real_t* data2 = (real_t*) malloc( mem_size);
  
  ///////////////////////////////////////////////////
  // compute GPU solution to 3D heat equation
  ///////////////////////////////////////////////////
  
  // inital condition
  initCondition3D (data1);

  // allocate device memory
  real_t* d_data1;
  real_t* d_data2;

  // device memory allocation (using cudaMallocPitch)
  size_t pitchBytes1;
  cudaMallocPitch((void**) &d_data1, &pitchBytes1, NX*sizeof(real_t), NY*NZ);
  unsigned int _pitch1 = pitchBytes1 / sizeof(real_t);

  size_t pitchBytes2; // should be the same as pitchBytes1
  cudaMallocPitch((void**) &d_data2, &pitchBytes2, NX*sizeof(real_t), NY*NZ);
  //unsigned int _pitch2 = pitchBytes2 / sizeof(real_t);

  // copy host memory to device
  cudaMemcpy2D(d_data1,pitchBytes1,
		data1  ,NX*sizeof(real_t), NX*sizeof(real_t), NY*NZ,
		cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_data2,pitchBytes2,
		data1  ,NX*sizeof(real_t), NX*sizeof(real_t), NY*NZ,
		cudaMemcpyHostToDevice);
 
  // setup execution parameters for cuda kernel
  // grid dimension for sharedmem2 kernel
  dim3  threads;
  dim3  grid;
  if (useOrder2) {
    threads.x=BLOCK_DIMX;
    threads.y=BLOCK_DIMY;
    grid.x   = (NX+BLOCK_INNER_DIMX-1)/BLOCK_INNER_DIMX;
    grid.y   = (NY+BLOCK_INNER_DIMY-1)/BLOCK_INNER_DIMY;
  } else { // 4th order
    // threads.x=BLOCK_DIMX2;
    // threads.y=BLOCK_DIMY2;
    // grid.x   = (NX+BLOCK_INNER_DIMX2-1)/BLOCK_INNER_DIMX2;
    // grid.y   = (NY+BLOCK_INNER_DIMX2-1)/BLOCK_INNER_DIMY2;
  }
  printf("grid  size : %u %u\n",grid.x,grid.y);
  printf("block size : %u %u\n",threads.x,threads.y);
  
  // setup parameters (using device constant memory)
  cudaMemcpyToSymbol(::o2Gpu ,&o2 ,sizeof(struct SecondOrderParam),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(::o4Gpu, &o4, sizeof(struct FourthOrderParam),0,cudaMemcpyHostToDevice);

  // start timer
  gpuTimer.start();

  // time loop executing sharedmem2 kernel
  unsigned int iTime=0;
  int iOutput = -1;
  for (real_t t=0.0f; t<TMAX; t+=(4*DT), iTime+=4) {
    
    if (useOrder2) {
      
      heat3d_ftcs_sharedmem4_order2_kernel<<< grid, threads >>>( d_data1, 
								 d_data2, 
								 _pitch1, 
								 NX, NY, NZ);
      
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
      heat3d_ftcs_sharedmem4_order2_kernel<<< grid, threads >>>( d_data2, 
								 d_data1, 
								 _pitch1,
								 NX, NY, NZ);
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
    } else {
      printf("Kernel for order 4 is not implemented !");
      // heat3d_ftcs_sharedmem2_order4_kernel<<< grid, threads >>>( d_data1, 
      // 								 d_data2, 
      // 								 _pitch1, 
      // 								 NX, NY, NZ);
      
      // // check if kernel execution generated and error
      // getLastCudaError("Kernel execution failed");
      
      // heat3d_ftcs_sharedmem2_order4_kernel<<< grid, threads >>>( d_data2, 
      // 								 d_data1, 
      // 								 _pitch1,
      // 								 NX, NY, NZ);
      // // check if kernel execution generated and error
      // getLastCudaError("Kernel execution failed");
    }
    
    
    if (ENABLE_GPU_SAVE) {
      
      if (iTime%T_OUTPUT == 0) {
	iOutput++;
	checkCudaErrors( cudaMemcpy2D( data1, NX*sizeof(real_t),
				       d_data1, pitchBytes1, 
				       NX*sizeof(real_t), NY*NZ,
				       cudaMemcpyDeviceToHost) );
      }
      
      // VTK output
      if (SAVE_VTK and iTime%T_OUTPUT == 0)
	save_vtk(data1, "heat3d_gpu_shmem4_",iOutput);
      
      // HDF5 output
      if (SAVE_HDF5 and iTime%T_OUTPUT == 0)
	save_hdf5(data1, "heat3d_gpu_shmem4_",iOutput);

    }
  }
  
  
  // stop timer
  gpuTimer.stop();
  real_t gpu_time = gpuTimer.elapsed();
  printf( "GPU Processing time: %f (s)\n", gpu_time);
  
  // copy result from device to host
  real_t *resGPU = (real_t*) malloc( mem_size);
  cudaMemcpy2D( resGPU,  NX*sizeof(real_t),
		d_data1, pitchBytes1     , NX*sizeof(real_t), NY*NZ,
		cudaMemcpyDeviceToHost);
  
  if (SAVE_HDF5)
    write_xdmf_wrapper("heat3d_gpu_shmem4",N_ITER,T_OUTPUT);
  
  ////////////////////////////////////////////////////////
  // compute reference (CPU) solution to 3D heat equation
  // for performance comparison
  ////////////////////////////////////////////////////////
  printf("compute CPU reference solution\n");
  initCondition3D (data1);
  initCondition3D (data2);
  
  Timer cpuTimer;
  cpuTimer.start();
  
  // time loop
  iTime=0;
  for (real_t t=0.0f; t<TMAX; t+=(4*DT), iTime+=4) {
    
    // compute next 2 time steps
    if (useOrder2) {
      heat3d_ftcs_cpu_order2( data1, data2);
      heat3d_ftcs_cpu_order2( data2, data1);
      heat3d_ftcs_cpu_order2( data1, data2);
      heat3d_ftcs_cpu_order2( data2, data1);
    } else {
      heat3d_ftcs_cpu_order4( data1, data2);
      heat3d_ftcs_cpu_order4( data2, data1);
      heat3d_ftcs_cpu_order4( data1, data2);
      heat3d_ftcs_cpu_order4( data2, data1);
    }
  }
  
  // stop timer
  cpuTimer.stop();
  real_t cpu_time = cpuTimer.elapsed();
  
  printf( "CPU Processing time: %g (s)\n", cpu_time);
  printf( "Speedup GPU/CPU : %f\n",cpu_time/gpu_time);

  printf("...comparing the results\n");
  double sum = 0, delta = 0;
  for(unsigned i = 0; i < NX*NY*NZ; i++){
    delta += (resGPU[i] - data1[i]) * (resGPU[i] - data1[i]);
    sum   += data1[i] * data1[i];
  }
  double L2norm = sqrt(delta / sum);
  printf("iteration %d relative L2 norm: %.10g\n", iTime, L2norm);

 
  // cleanup memory
  free(data1);
  free(data2);
  free(resGPU);
  
  checkCudaErrors(cudaFree(d_data1));
  checkCudaErrors(cudaFree(d_data2));
  
  cudaDeviceSynchronize();
  cudaDeviceReset();

  exit(0);
}
