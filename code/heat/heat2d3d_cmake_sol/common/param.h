/**
 * \file param.h
 * \brief Definition of all parameters for a 2D/3D Heat Equation solver.
 *
 * \note Patameters with attribute __constant__  are aimed to be used
 * inside a CUDA kernel.
 *
 * \date 17-dec-2009
 */
#ifndef PARAM_H_
#define PARAM_H_

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float  real_t;
#endif // USE_DOUBLE


#include <cmath>
#include <string>

// M_PI is defined in math.h but not in cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI

/**
 * Second order scheme parameters
 */
struct SecondOrderParam {

  real_t R;
  real_t R2;
  real_t R2b;
  real_t R3;
  real_t R3b;

}; // SecondOrderParam

extern struct SecondOrderParam o2;
#ifdef __CUDACC__
__constant__ struct SecondOrderParam o2Gpu;
#endif

/**
 * Fourth order scheme parameters
 */
struct FourthOrderParam {

  real_t S;
  real_t S2;
  real_t S3;

}; // FourthOrderParam

extern struct FourthOrderParam o4;
#ifdef __CUDACC__
__constant__ struct FourthOrderParam o4Gpu;
#endif

/*
 * Some useful global constants.
 */

/**@{ spatial domain sizes */
extern real_t LX;
extern real_t LY;
extern real_t LZ;
/**@} */

/**@{ spatial domain discrete sizes */
extern unsigned int NX;
extern unsigned int NY;
extern unsigned int NZ;
/**@} */

extern real_t CFL; /**< Courant-Friedrichs-Lewy number */
extern real_t DX;  /**< X resolution */
extern real_t DY;  /**< Y resolution */
extern real_t DZ;  /**< Z resolution */

extern std::string   PROBLEM;  /**< problem name used in init routine */
extern real_t        DT;       /**< delta t */
extern unsigned int  N_ITER;   /**< number of iterations */
extern real_t        TMAX;     /**< maximum time */
extern unsigned int  T_OUTPUT; /**< number of time step between outputs */

extern real_t         ALPHA;  /**< Diffusion coefficient */

extern bool useOrder2;    /**< use the second order scheme */
extern bool useOrder2b;   /**< use the second order scheme with a 3x3 stencil */


/**@{ output options */
extern bool SAVE_MGL;
extern bool SAVE_PGM;
extern bool SAVE_VTK;
extern bool SAVE_HDF5;
extern bool ENABLE_GPU_SAVE; /**< for debugging the GPU version */
/**@} */

/** read parameter file and initialize global constants */
void readParamFile(const std::string &filename);

/** print parameters */
void printParameters(const char* header);

#endif // PARAM_H_
