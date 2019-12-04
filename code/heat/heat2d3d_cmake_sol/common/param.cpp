/**
 * \file param.cpp
 * \brief Implementation of all parameters for a 2D Heat Equation solver.
 *
 * \date 17-dec-2009
 */
#include "param.h"
#include "GetPot.h"

/*
 * Initialize global constants.
 */
void readParamFile(const std::string &filename)
{
  GetPot config(filename.c_str());
  
  /* geometry */
  LX = config("geometry/LX",1.0);
  LY = config("geometry/LY",1.0);
  LZ = config("geometry/LZ",1.0);

  NX = config("geometry/NX",256);
  NY = config("geometry/NY",256);
  NZ = config("geometry/NZ",1);
  
  DX = LX/(NX-1);
  DY = LY/(NY-1);
  DZ = NZ!=1 ? LZ/(NZ-1) : 0.0;

  /* initialization : problem name */
  PROBLEM = config("run/PROBLEM","square");

  /* run : number of iterations */
  N_ITER = config("run/N_ITER",100);

  /* time parameters */
  CFL   = config("scheme/CFL",0.125);
  DT    = CFL*DX*DX; 
  TMAX  = N_ITER*DT;
  T_OUTPUT= config("run/T_OUTPUT",20);

  /* diffusion coef */
  ALPHA = config("scheme/ALPHA",1.0);

  /* set scheme order */
  std::string tmp = config("scheme/useOrder2","yes");
  useOrder2 = (!tmp.compare("yes")) ? true : false;

  tmp = config("scheme/useOrder2b","yes");
  useOrder2b = (!tmp.compare("yes")) ? true : false;

  o2.R   = ALPHA*DT/DX/DX;
  o2.R2  = 1.0f-2*2*o2.R; // 2D
  o2.R2b = 1.0f-  9*o2.R;
  o2.R3  = 1.0f-3*2*o2.R; // 3D
  o2.R3b = 1.0f- 27*o2.R; // 3D

  o4.S   = o2.R/12.;
  o4.S2  = 1.0-2*30*o2.R/12.; // 2D
  o4.S3  = 1.0-3*30*o2.R/12.; // 3D

  /* output options */
  tmp = config("output/SAVE_PGM","yes"); 
  SAVE_PGM = (tmp == "yes") ? true : false; 

  tmp = config("output/SAVE_MGL","no"); 
  SAVE_MGL = (tmp == "yes") ? true : false; 

  tmp = config("output/SAVE_VTK","no");
  SAVE_VTK = (tmp == "yes") ? true : false;

  tmp = config("output/SAVE_HDF5","no");
  SAVE_HDF5 = (tmp == "yes") ? true : false;

  /* for debugging GPU version */
  tmp = config("output/ENABLE_GPU_SAVE","no");
  ENABLE_GPU_SAVE = (tmp == "yes") ? true : false;

#ifdef __CUDACC__
  // we need to copy scheme's param into GPU constant memory space
#endif

} // readParamFile

void printParameters(const char* header)
{
  printf("=======================\n");
  printf("%s\n",header);
  printf("=======================\n");
  printf("LX,LY,LZ : %f,%f,%f\n",LX,LY,LZ);
  printf("NX,NY,NZ : %d,%d,%d\n",NX,NY,NZ);
  printf("CFL      : %.10f\n",CFL);
  printf("DX,DY,DZ : %.10f,%.10f,%.10f\n",DX,DY,DZ);
  printf("DT       : %f\n",DT);
  printf("PROBLEM  : %s\n",PROBLEM.c_str());
  printf("TMAX     : %.10f\n",TMAX);
  printf("T_OUPUT  : %d\n",T_OUTPUT);
  printf("N_ITER   : %d\n",N_ITER);
  printf("ALPHA    : %.10f\n",ALPHA);
  printf("R        : %.10f\n",o2.R);
  printf("R2       : %.10f\n",o2.R2);
  printf("R2b      : %.10f\n",o2.R2b);
  printf("R3       : %.10f\n",o2.R3);
  printf("S        : %.10f\n",o4.S);
  printf("S2       : %.10f\n",o4.S2);
  printf("S3       : %.10f\n",o4.S3);


} // printParameters

real_t LX;
real_t LY;
real_t LZ;

unsigned int NX;
unsigned int NY;
unsigned int NZ;

real_t CFL;
real_t DX;
real_t DY;
real_t DZ;

std::string PROBLEM;
real_t DT;
unsigned int N_ITER;
real_t TMAX;
unsigned int T_OUTPUT;

real_t ALPHA;

struct SecondOrderParam o2;
struct FourthOrderParam o4;

bool useOrder2;
bool useOrder2b;

bool SAVE_MGL;
bool SAVE_PGM;
bool SAVE_VTK;
bool SAVE_HDF5;
bool ENABLE_GPU_SAVE;
