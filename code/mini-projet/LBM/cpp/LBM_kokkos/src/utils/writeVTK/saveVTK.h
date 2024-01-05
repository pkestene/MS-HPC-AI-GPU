#ifndef SAVE_VTK_H_
#define SAVE_VTK_H_

#include "lbm/LBMParams.h"

#include "lbm/real_type.h"

/**
 * Host routine to save data to file (vti = VTK image format).
 * 
 * CUDA device data array should be first copied into host memory,
 * before calling this routine.
 *
 * about VTK file format: ASCII, VtkImageData
 * Take care that VTK uses row major (i+j*nx)
 *
 * \param[in] rho
 * \param[in] ux
 * \param[in] uy
 * \param[in] params a LBMParams reference object (input only)
 * \param[in] outputVtkAscii true or false (use binary format)
 * \param[in] iStep integer, current time step used in filename
 *
 */
void saveVTK(const real_t* rho,
	     const real_t* ux,
	     const real_t* uy,
	     const LBMParams& params,
             bool outputVtkAscii,
	     int iStep);

#endif // SAVE_VTK_H_
