/**
 * \file heat2d_solver_gpu_shmem3.cu
 * \brief Solve 2D heat equation (finite difference method). GPU version (shmem3).
 *
 * We solve the 2D Heat equation \f$\partial_t \phi = \alpha \left[
 * \partial^2_x \phi + \partial^2_y \phi \right] \f$, \f$ 0 \leq x
 * \leq L_x \f$, \f$ 0 \leq y \leq L_y \f$, \f$ 0 \leq t\f$.\\
 *
 * Method : Finite Difference, FTCS scheme
 *
 * boundary condition : Dirichlet
 *
 * GPU version : shared memory 2
 *
 * \date 17-dec-2009.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h> // gettimeofday
#include <assert.h>   // assert
// includes, project
#include <cutil_inline.h>

// parameters
#include "param.h"

// for output results
#include "output.h"

// GPU solver
#include "heat2d_kernel_gpu_shmem3.cu"

// CPU solver
#include "heat_kernel_cpu.h"

// initial conditions
#include "misc.h"

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

  cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv) 
{
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
    cutilDeviceInit(argc, argv);
  else
    cudaSetDevice( cutGetMaxGflopsDeviceId() );

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
  printParameters("HEAT 2D - GPU (SHMEM3)");





  unsigned int timer = 0;
  cutilCheckError( cutCreateTimer( &timer));

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

  // device memory allocation (using cudaMallocPitch)
//   size_t pitchBytes1;
//   cudaMallocPitch((void**) &d_data1, &pitchBytes1, NX*sizeof(real_t), NY);
//   unsigned int _pitch1 = pitchBytes1 / sizeof(real_t);

//   size_t pitchBytes2; // should be the same as pitchBytes1
//   cudaMallocPitch((void**) &d_data2, &pitchBytes2, NX*sizeof(real_t), NY);
//   unsigned int _pitch2 = pitchBytes2 / sizeof(real_t);

  int _pitch1 = NX;
  //int _pitch2 = NX;
  cutilSafeCall( cudaMalloc( (void**) &d_data1, mem_size));
  cutilSafeCall( cudaMalloc( (void**) &d_data2, mem_size));
  
  // copy host memory to device (using cudaMemcpy2D)
  cutilSafeCall( cudaMemcpy( d_data1, data1, mem_size, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy( d_data2, data1, mem_size, cudaMemcpyHostToDevice) );


  // setup execution parameters for cuda kernel
  // grid dimension for sharedmem kernel
  assert( NX % (ROWS_NBLOCKS * ROWS_BLOCK_X) == 0 );
  assert( NY % ROWS_BLOCK_Y == 0 );
  dim3 grid_rows(NX / (ROWS_NBLOCKS * ROWS_BLOCK_X), NY / ROWS_BLOCK_Y);
  dim3 threads_rows(ROWS_BLOCK_X, ROWS_BLOCK_Y);
    
  assert( NX % COLS_BLOCK_X == 0);
  assert( NY % (COLS_NBLOCKS * COLS_BLOCK_Y) == 0);
  dim3 grid_cols( NX / COLS_BLOCK_X, NY / (COLS_NBLOCKS *COLS_BLOCK_Y));
  dim3 threads_cols(COLS_BLOCK_X, COLS_BLOCK_Y);

  printf("rows grid  size : %u %u\n",grid_rows.x,grid_rows.y);
  printf("rows block size : %u %u\n",threads_rows.x,threads_rows.y);
  printf("cols grid  size : %u %u\n",grid_cols.x,grid_cols.y);
  printf("cols block size : %u %u\n",threads_cols.x,threads_cols.y);
  
  // setup parameters (using device constant memory)
  cudaMemcpyToSymbol(::o2Gpu,&o2,sizeof(real_t),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(::o4Gpu,&o4,sizeof(real_t),0,cudaMemcpyHostToDevice);

  // start timer
  cutilSafeCall( cudaDeviceSynchronize() );
  cutilCheckError( cutResetTimer( timer));
  cutilCheckError( cutStartTimer( timer));

  // time loop executing sharedmem kernel
  int iTime   =  0;
  int iOutput = -1;
  for (real_t t=0.0f; t<TMAX; t+=(2*DT), iTime+=2) {
    heat2d_ftcs_sharedmem3_row_kernel<<< grid_rows, threads_rows >>>( d_data1, d_data2, 
								      _pitch1, NX, NY);
    cutilCheckMsg("Kernel execution failed");
    heat2d_ftcs_sharedmem3_col_kernel<<< grid_cols, threads_cols >>>( d_data1, d_data2, 
								      _pitch1, NX, NY);

    heat2d_ftcs_sharedmem3_row_kernel<<< grid_rows, threads_rows >>>( d_data2, d_data1, 
								      _pitch1, NX, NY);
    heat2d_ftcs_sharedmem3_col_kernel<<< grid_cols, threads_cols >>>( d_data2, d_data1, 
								      _pitch1, NX, NY);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    /* save output (just for cross-checking, do not save when
       measuring computing time */
    if (ENABLE_GPU_SAVE) {
      
      if (iTime%T_OUTPUT == 0) {
	iOutput++;
	cutilSafeCall( cudaMemcpy( data1, d_data1, NX*NY*sizeof( real_t),
				   cudaMemcpyDeviceToHost) );
      }
            
      // PGM save
      if (SAVE_PGM and iTime%T_OUTPUT == 0)
	save_pgm(data1, "heat2d_gpu_shmem3_",iOutput,NX,NY);
      
      // MathGL save (3D view)
      if (SAVE_MGL and iTime%T_OUTPUT == 0)
	save_mgl(data1, "heat2d_gpu_shmem3_",iOutput,NX,NY);

      // VTK output
      if (SAVE_VTK and iTime%T_OUTPUT == 0)
	save_vtk(data1, "heat2d_gpu_shmem3_",iOutput);

      // HDF5 output
      if (SAVE_HDF5 and iTime%T_OUTPUT == 0)
	save_hdf5(data1, "heat2d_gpu_shmem3_",iOutput);
      
    }

  }
    
  // stop timer
  cutilSafeCall(   cudaDeviceSynchronize());
  cutilCheckError( cutStopTimer( timer));
  real_t gpu_time = cutGetTimerValue( timer);
  
  // copy result from device to host
  real_t *resGPU = (real_t*) malloc( mem_size);
  cudaMemcpy( resGPU, d_data1, NX*NY*sizeof( real_t), cudaMemcpyDeviceToHost);
  
  printf( "GPU Processing time: %f (ms)\n", gpu_time);
  cutilCheckError( cutResetTimer( timer));
  
  if (SAVE_HDF5)
    write_xdmf_wrapper("heat2d_gpu_shmem3",N_ITER,T_OUTPUT);

  ////////////////////////////////////////////////////////
  // compute reference (CPU) solution to 2D heat equation
  // for performance comparison
  ////////////////////////////////////////////////////////
  initCondition2D (data1);
  initCondition2D (data2);

  cutilSafeCall( cudaDeviceSynchronize() );
  cutilCheckError( cutStartTimer( timer));
  
  //timeval_t start,stop;
  //start = get_time();
  // time loop
  iTime=0;
  for (real_t t=0.0f; t<TMAX; t+=(2*DT), iTime+=2) {
    
    // compute next 2 time steps
    heat2d_ftcs_cpu_order2( data1, data2);
    heat2d_ftcs_cpu_order2( data2, data1);
  }
  
  cutilSafeCall( cudaDeviceSynchronize() );
  cutilCheckError( cutStopTimer( timer));
  real_t cpu_time = cutGetTimerValue( timer);
  
  printf( "CPU Processing time: %g (ms)\n", cpu_time);
  printf( "Speedup GPU/CPU : %f\n",cpu_time/gpu_time);

  cutilCheckError( cutDeleteTimer( timer));
   
  printf("...comparing the results\n");
  double sum = 0, delta = 0;
  for(unsigned i = 0; i < NX*NY; i++){
    delta += (resGPU[i] - data1[i]) * (resGPU[i] - data1[i]);
    sum   += data1[i] * data1[i];
  }
  double L2norm = sqrt(delta / sum);
  printf("iteration %d relative L2 norm: %E\n", iTime, L2norm);

 
  // cleanup memory
  free(data1);
  free(data2);
  free(resGPU);
  
  cutilSafeCall(cudaFree(d_data1));
  cutilSafeCall(cudaFree(d_data2));
  
  cudaDeviceReset();
}
