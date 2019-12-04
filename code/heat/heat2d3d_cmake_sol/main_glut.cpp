// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, application
#include "param.h"
#include "output.h"
#include "misc.h"

// CPU solver
#include "heat_kernel_cpu.h"

// GPU solver
#ifdef __CUDACC__
#include "heat2d_kernel_gpu_naive.cu"
#endif // __CUDACC__

/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GL/glew.h>
#include <GL/freeglut.h> // glutLeaveMainLoop is declared in freeglut_ext.h
#include <GL/glext.h>
#include <GL/glx.h>

// cuda SDK includes
#ifdef __CUDACC__
#include <helper_functions.h>
#include "CudaTimer.h"
#include "Timer.h"

// cuda helper
#include "cuda_helper.cu"

#define CUDA_SAFE_CALL_NO_SYNC( call) {					\
    cudaError err = call;						\
    if( cudaSuccess != err) {						\
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, cudaGetErrorString( err) );		\
      exit(EXIT_FAILURE);						\
    } }

#define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#define CUDA_SAFE_THREAD_SYNC( ) {					\
    cudaError err = cudaDeviceSynchronize();				\
    if ( cudaSuccess != err) {						\
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
	      __FILE__, __LINE__, cudaGetErrorString( err) );		\
    } }

#endif // __CUDACC__

// OpenGL pixel buffer object and texture id's
GLuint gl_PBO, gl_Tex;

// cuda resource for OpenGL interoperability
#ifdef __CUDACC__
struct cudaGraphicsResource* cuda_PBO;
dim3 grid;
dim3 threads;
#endif // __CUDACC__

void createTexture();
void deleteTexture();
void createPBO();
void deletePBO();

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif // __CUDACC__


// make data1, data2 global pointers so that they can be reached in
// glut callback function 
real_t* data1;
real_t* data2;
int   * mask;
unsigned int mem_size;

// GPU device memory
real_t* d_data1;
real_t* d_data2;
int   * d_mask;

// for display
unsigned char* pixels;
unsigned char* pixels_pbo; // for CUDA

int nbStepToCompute = 1;
int totalTimeSteps = 0;
bool animate = false;

// window sizes
int width, height;

// mouse clicked position
int ipos_old,jpos_old;
int mask_flag;


/*
 * data buffer to pixel buffer conversion
 */

/* ============================= */
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
unsigned char value( float n1, float n2, int hue ) {
  if (hue > 360)      hue -= 360;
  else if (hue < 0)   hue += 360;
  
  if (hue < 60)
    return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
  if (hue < 180)
    return (unsigned char)(255 * n2);
  if (hue < 240)
    return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
  return (unsigned char)(255 * n1);
}

/* ============================= */
#ifdef __CUDACC__
__global__ void data_to_pixel_kernel( const real_t *dataIn,
				      unsigned char *pixelOut,
				      int nx)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * nx;

    real_t l = dataIn[offset];
    real_t s = 1;
    int h = (180 + (int)(360.0f * dataIn[offset])) % 360;
    real_t m1, m2;

    if (l <= 0.5f)
      m2 = l * (1 + s);
    else
      m2 = l + s - l * s;
    m1 = 2 * l - m2;

    pixelOut[offset*4 + 0] = value( m1, m2, h+120 );
    pixelOut[offset*4 + 1] = value( m1, m2, h );
    pixelOut[offset*4 + 2] = value( m1, m2, h-120 );
    pixelOut[offset*4 + 3] = 255;
    
}
#endif // __CUDACC__

/* ============================= */
void data_to_pixel(real_t* data, 
		   unsigned char *pixelOut, 
		   int nx, int ny)
{
#ifdef __CUDACC__
  // GPU version
  
  CUDA_SAFE_THREAD_SYNC( );

  // For plotting, map the gl_PBO pixel buffer into CUDA context
  // space, so that CUDA can modify the device pointer plot_rgba_pbo
  checkCudaErrors( cudaGraphicsMapResources(1, &cuda_PBO, NULL));

  size_t num_bytes; 
  checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&pixels_pbo, &num_bytes, cuda_PBO));

  // call CUDA kernel computing colors
  {
    dim3    threads_pixels(16,16);
    dim3    blocks_pixels( (NX+threads_pixels.x-1)/threads_pixels.x,
			   (NY+threads_pixels.y-1)/threads_pixels.y );
    data_to_pixel_kernel<<<blocks_pixels,threads_pixels>>>( d_data1,
							    pixels_pbo,
							    NX);

  }

  // unmap resources so that OpenGL can display 
  checkCudaErrors( cudaGraphicsUnmapResources(1, &cuda_PBO, NULL));

#else
  // CPU version
  for (int j=0; j<ny; j++)
    for (int i=0; i<nx; i++) {

      int offset = i+nx*j;
      real_t l = data[offset];
      real_t s = 1;
      int h = (180 + (int)(360.0f * data[offset])) % 360;
      real_t m1, m2;
      
      if (l <= 0.5f)
        m2 = l * (1 + s);
      else
        m2 = l + s - l * s;
      m1 = 2 * l - m2;
      
      pixelOut[offset*4 + 0] = value( m1, m2, h+120 );
      pixelOut[offset*4 + 1] = value( m1, m2, h );
      pixelOut[offset*4 + 2] = value( m1, m2, h -120 );
      pixelOut[offset*4 + 3] = 255;

    }
#endif
}

/*
 * GLUT callback functions
 */

/* ============================= */
void callback_keyboard(unsigned char key, int x, int y)
{
  switch (key) {

  case 'q': case 27: /* ESCAPE key */
    glutLeaveMainLoop () ; // exit program
    break;

  case 32 : /* keyboard SPACE: Start/Stop the animation */
    animate= not animate;
    if (animate) {
      printf("restarting simulations from step : %d\n",totalTimeSteps);
    }
    break ;

  case 's' : case 'S' : /* Do only a single step */
    {
      animate=false;
      nbStepToCompute=1;
    }
    break;
    
  case 'r' :  /* restart */
    {
      printf("Restart run\n");
      initCondition2D(data1);
      initCondition2D(data2);
#ifdef __CUDACC__
      checkCudaErrors( cudaMemcpy( d_data1, data1, mem_size,
				  cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy( d_data2, data1, mem_size,
			      cudaMemcpyHostToDevice) );
#endif // __CUDACC__
    }
    break;

  case 'R' : /* also re-init mask */
    {
      printf("Restart run; reset mask\n");
      initCondition2D(data1);
      initCondition2D(data2);
      for (unsigned int i=0; i<NX*NY; i++)
	mask[i]=1.0;
#ifdef __CUDACC__
      checkCudaErrors( cudaMemcpy( d_data1, data1, mem_size,
				  cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy( d_data2, data1, mem_size,
			      cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy( d_mask, mask, sizeof(int)*NX*NY,
			      cudaMemcpyHostToDevice) );
#endif // __CUDACC__
    }
    break;

  default:
    fprintf(stderr, "Unused key\n");
    break;
  }
    
} // callback_keyboard

/* ============================= */
void computeTimeStep()
{

  // update data with our Heat Equation solver
  for (int i=0; i<10; i++)  {

#ifdef __CUDACC__

    if (useOrder2) { // use the 2nd order accurate scheme
      
      heat2d_ftcs_naive_order2_mask_kernel<<< grid, threads >>>( d_data1, d_data2, d_mask,
								 NX, NY,
								 o2.R, o2.R2);
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
      heat2d_ftcs_naive_order2_mask_kernel<<< grid, threads >>>( d_data2, d_data1, d_mask,
								 NX, NY,
								 o2.R, o2.R2);   
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
    } else if (useOrder2b) { // use the 2nd order accurate scheme
      
      heat2d_ftcs_naive_order2b_kernel<<< grid, threads >>>( d_data1, d_data2, 
    							     NX, NY,
    							     o2.R, o2.R2b);
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
      heat2d_ftcs_naive_order2b_kernel<<< grid, threads >>>( d_data2, d_data1, 
    							     NX, NY,
    							     o2.R, o2.R2b);   
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");

    } else { // use the 4th order accurate scheme
      
      printf("Masked kernel for 4th order is not implemented; TODO\n");

      heat2d_ftcs_naive_order4_kernel<<< grid, threads >>>( d_data1, d_data2, 
    							    NX, NY,
    							    o4.S, o4.S2);
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
      heat2d_ftcs_naive_order4_kernel<<< grid, threads >>>( d_data2, d_data1, 
    							     NX, NY,
    							     o4.S, o4.S2);   
      // check if kernel execution generated and error
      getLastCudaError("Kernel execution failed");
      
    }

#else
      // compute time step
      if (useOrder2) {
	heat2d_ftcs_cpu_order2_with_mask(data1, data2, mask);
	heat2d_ftcs_cpu_order2_with_mask(data2, data1, mask);
      } else if (useOrder2b) {
	heat2d_ftcs_cpu_order2b(data1, data2);
	heat2d_ftcs_cpu_order2b(data2, data1);
      } else {
	heat2d_ftcs_cpu_order4(data1, data2);
	heat2d_ftcs_cpu_order4(data2, data1);      
      }
#endif // __CUDACC__

      totalTimeSteps += 2;
  } // end for loop
}

/* ============================= */
void renderToScreen()
{
  // fill pixels
  data_to_pixel(data1,pixels,NX,NY);

#ifndef __CUDACC__
  // this is only true for the CPU version

  // Fill the pixel buffer object with the plot_rgba array
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,NX*NY*4,
	       (void **)pixels,GL_STREAM_COPY);
#endif // ! __CUDACC__

  // Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,NX,NY,GL_RGBA,GL_UNSIGNED_BYTE,0);
  
  // Render one quad to the screen and colour it using our texture
  // i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_QUADS);
  glTexCoord2f (0.0, 0.0); glVertex3f (0.0 , 0.0 , 0.0);
  glTexCoord2f (1.0, 0.0); glVertex3f (NX  , 0.0 , 0.0);
  glTexCoord2f (1.0, 1.0); glVertex3f (NX  , NY  , 0.0);
  glTexCoord2f (0.0, 1.0); glVertex3f (0.0 , NY  , 0.0);
  glEnd();
  
  glutSwapBuffers();
  glutReportErrors();

}

/* ============================= */
void callback_display(void)
{

  if (animate) {
    
    // compute and display continuously
    computeTimeStep();
    renderToScreen();

  } else {
    
    // only compute/display a finite number of steps
    if (nbStepToCompute) {
      computeTimeStep();
      renderToScreen();
      nbStepToCompute--;
    }
    
  }
 
}

/* ============================= */
void callback_idle(void)
{

  callback_display();

}

/* ============================= */
void callback_resize(int w, int h)
{
  width = w;
  height = h;
  glViewport (0, 0, w, h);
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  glOrtho (0., NX, 0., NY, -200. ,200.);
  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();
}

/* ============================= */
void callback_mouse(int button, int state,
		    int x, int y)
{
  // do something when mouse button is clicked !

  float xx,yy;

  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
    xx=x;
    yy=y;
    ipos_old=xx/width*NX;
    jpos_old=(height-yy)/height*NY;
    //mask[ipos_old+jpos_old*NX] = 0;
    mask_flag = 0;
  }

  if ((button == GLUT_RIGHT_BUTTON) && (state == GLUT_DOWN)) {
    xx=x;
    yy=y;
    ipos_old=xx/width*NX;
    jpos_old=(height-yy)/height*NY;
    //mask[ipos_old+jpos_old*NX] = 1;
    mask_flag = 1;
  }

  // redisplay
  //glutPostRedisplay();

} // callback_mouse

/* ============================= */
void callback_motion(int x, int y)
{
  // derived from LB_Demo

  // GLUT call back for when the mouse is moving
  // This sets the mask array value to mask_flag as set in the mouse callback
  // It will draw a staircase line if we move more than one pixel since the
  // last callback - that makes the coding a bit cumbersome:
  float xx,yy,frac;
  int ipos,jpos,i,j,i1,i2,j1,j2, jlast, jnext;
  xx=x;
  yy=y;
  ipos=(int)(xx/width*(float)NX);
  jpos=(int)((height-yy)/height*(float)NY);
  
  if (ipos <= ipos_old){
    i1 = ipos;
    i2 = ipos_old;
    j1 = jpos;
    j2 = jpos_old;
  }
  else {
    i1 = ipos_old;
    i2 = ipos;
    j1 = jpos_old;
    j2 = jpos;
  }
  
  jlast=j1;
  
  for (i=i1;i<=i2;i++){
    if (i1 != i2) {
      frac=(float)(i-i1)/(float)(i2-i1);
      jnext=(int)(frac*(j2-j1))+j1;
    }
    else {
      jnext=j2;
    }
    if (jnext >= jlast) {
      mask[i+NX*jlast]=mask_flag;
      for (j=jlast; j<=jnext; j++){
	mask[i+NX*j]=mask_flag;
      }
    }
    else {
      mask[i+NX*jlast]=mask_flag;
      for (j=jnext; j<=jlast; j++){
	mask[i+NX*j]=mask_flag;
      }
    }
    jlast = jnext;
  }

#ifdef __CUDACC__
  // Copy the mask array (host) to d_mask array (device)
  checkCudaErrors(cudaMemcpy((void *)d_mask, (void *)mask,
			    sizeof(int)*NX*NY, cudaMemcpyHostToDevice) );    
 
#endif // __CUDACC__
  
  
  ipos_old=ipos;
  jpos_old=jpos;
  
} // callback_motion

/* ============================= */
void cleanup()
{

  free(data1);
  free(data2);
  free(mask);
  free(pixels);

#ifdef __CUDACC__
  checkCudaErrors( cudaFree(d_data1) );
  checkCudaErrors( cudaFree(d_data2) );
  checkCudaErrors( cudaFree(d_mask) );
#endif // __CUDACC__

  deleteTexture();
  deletePBO();

  printf("leaving GLUT gui...\n");
}

// OpenGL specific data construtor/destructor
void createTexture(void)
{

  if(gl_Tex) deleteTexture();
  
  // Enable Texturing
  glEnable(GL_TEXTURE_2D);
  
  // Generate a texture identifier
  glGenTextures(1, &gl_Tex);
  
  // Make this the current texture (remember that GL is state-based)
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  
  // Allocate the texture memory. 
  // The last parameter is NULL since we only want to allocate memory,
  // not initialize it
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NX, NY, 0, 
	       GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  
  // texture properties:
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call
  
  printf("Texture created...\n");

}

void deleteTexture()
{
  if (gl_Tex) {
    glDeleteTextures(1, &gl_Tex);
    gl_Tex = 0;
  }
  
  printf("Texture deleted...");

}

void createPBO()
{
  // Create pixel buffer object and bind to gl_PBO. We store the data we want to
  // plot in memory on the graphics card - in a "pixel buffer". We can then 
  // copy this to the texture defined above and send it to the screen

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffersARB(1,&gl_PBO);
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);

#ifdef __CUDACC__
  // Reserve memory space for PBO data (see plot_rgba)
  //glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, (nxg)*(nyg)*sizeof(unsigned int), NULL, GL_STREAM_COPY_ARB);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, NX*NY*sizeof(real_t), NULL, GL_STREAM_COPY_ARB);

  // Is the following line really necessary ???
  //glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  cudaGraphicsGLRegisterBuffer( &cuda_PBO, gl_PBO, cudaGraphicsMapFlagsNone );

#endif // __CUDACC__  
  
  printf("PBO created...\n");

}

void deletePBO()
{
  if(gl_PBO) {
    // delete the gl_PBO

    // unmap CUDA resources
#ifdef __CUDACC__
    //checkCudaErrors( cudaGraphicsUnregisterResource( cuda_PBO ) );
    //checkCudaErrors( "cudaGraphicsUnRegisterResource failed");
#endif // __CUDACC__
    
    glDeleteBuffersARB( 1, &gl_PBO );
    gl_PBO=0;

#ifdef __CUDACC__
    cuda_PBO=NULL;
#endif // __CUDACC__
    
  }

  printf("PBO deleted...");

}


//////////////////////////////////////////////////////////
// avoid linking problem with pthread
void junk() {
  int i;
  (void) i;
  i=pthread_getconcurrency();
};
//////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
//////////////////////////////////////////////////////////
{

#ifdef __CUDACC__
  // first, find a CUDA device and set it to graphic interop
  // we only require that the chosen device to have hardware >= 1.0
  cudaDeviceProp  prop;
  int dev;
  memset( &prop, 0, sizeof( cudaDeviceProp ) );
  prop.major = 1;
  prop.minor = 0;
  checkCudaErrors( cudaChooseDevice( &dev, &prop ) );
  cudaGLSetGLDevice( dev );
#endif  

  // default parameter file
  std::string paramFile("heatEqSolver.par");

  // if argv[1] exists use it as a parameter file
  if (argc>1) {
    printf("trying to read parameters from file %s ...\n",argv[1]);
    paramFile = std::string(argv[1]);
  }

  // read parameter file
  readParamFile(paramFile);

  if (NZ!=1) {
    fprintf(stderr, "You must must set NZ=1 !!!; I will use NZ=1 from now on.\n");
    NZ=1;
  }
    

  // print parameters
#ifdef __CUDACC__
  printParameters("HEAT solver on GPU");
#else
  printParameters("HEAT solver on CPU");
#endif

  // init GLUT GUI
  width=512;
  height=512;
  
  // allocate memory for drawing buffer, each pixels stores 4 char
  // values (R, G, B, A)
  pixels = (unsigned char*) malloc(NX * NY * 4);
  
  int c=1;
  char* dummy = strdup("");
  glutInit( &c, &dummy );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
  glutInitWindowSize( width, height );
#ifdef __CUDACC__
  glutCreateWindow( "Heat solver GUI - GPU" );
#else
  glutCreateWindow( "Heat solver GUI - CPU" );
#endif 

  /*
   * Check for OpenGL extension support 
   */
  printf("Loading GLEW extensions: %s\n", glewGetErrorString(glewInit()));
  if(!glewIsSupported(
   		      "GL_VERSION_2_0 " 
   		      "GL_ARB_pixel_buffer_object "
   		      "GL_EXT_framebuffer_object "
   		      )){
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    //return;
  }
  
  // create OpenGL resources
  createTexture();
  createPBO();

  // setup GLUT gui callbacks
  glutKeyboardFunc ( callback_keyboard );
  glutDisplayFunc  ( callback_display  );
  glutMouseFunc    ( callback_mouse    );
  glutMotionFunc   ( callback_motion   ); 
  glutReshapeFunc  ( callback_resize   );
  glutIdleFunc     ( callback_idle     );




  // use NZ=1 to do 2D simulations
  mem_size = sizeof(real_t) * NX * NY * NZ;
  
  // allocate host memory
  data1 = (real_t*) malloc(mem_size);
  data2 = (real_t*) malloc(mem_size);
  mask  = (int   *) malloc(sizeof(int) * NX * NY * NZ);
  
  // initalize the memory
  if (NZ==1) {
    initCondition2D(data1);
    initCondition2D(data2);
  } else { 
    initCondition3D(data1);
    initCondition3D(data2);
  }
  
  for (unsigned int i=0; i<NX*NY*NZ; i++)
    mask[i] = 1;
  
#ifdef __CUDACC__
  // GPU memory allocation
  checkCudaErrors( cudaMalloc( (void**) &d_data1, mem_size));
  checkCudaErrors( cudaMalloc( (void**) &d_data2, mem_size));
  checkCudaErrors( cudaMalloc( (void**) &d_mask , sizeof(int)*NX*NY));
  // copy host memory to device
  checkCudaErrors( cudaMemcpy( d_data1, data1, mem_size,
			      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy( d_data2, data1, mem_size,
			      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy( d_mask, mask, sizeof(int)*NX*NY,
			      cudaMemcpyHostToDevice) );
  
  // setup execution parameters for cuda kernel
  // grid dimension for naive kernel
  unsigned int threadsPerBlockX=16;
  unsigned int threadsPerBlockY=16;
  threads.x = threadsPerBlockX;
  threads.y = threadsPerBlockY;
  grid.x = (NX+threads.x-1)/threads.x;
  grid.y = (NY+threads.y-1)/threads.y;
#endif // __CUDACC__

  printf("Press SPACE to start/stop simulation\n");

  // register cleanup routine (executed when program exits)
  atexit(cleanup);

  // start GUI event loop (only ends if glutLeaveMainLoop is called is
  // a callback routine)
  glutMainLoop();

} // end main
