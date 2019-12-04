#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"

// ======================================================
// ======================================================
void macroscopic(const LBMParams& params, 
                 const velocity_array_t v,
                 const real_t* fin,
                 real_t* rho,
                 real_t* ux,
                 real_t* uy)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int base_index = i + nx*j; 

      real_t rho_tmp = 0;
      real_t ux_tmp  = 0;
      real_t uy_tmp  = 0;
      for (int ipop = 0; ipop < npop; ++ipop) {
        
        int index = base_index + ipop*nx*ny;

        // Oth order moment
        rho_tmp +=             fin[index];

        // 1st order moment
        ux_tmp  += v(ipop,0) * fin[index];
        uy_tmp  += v(ipop,1) * fin[index];

      } // end for ipop

      rho[base_index] = rho_tmp;
      ux[base_index]  = ux_tmp/rho_tmp;
      uy[base_index]  = uy_tmp/rho_tmp;

    } // end for i
  } // end for j

} // macroscopic

// ======================================================
// ======================================================
void equilibrium(const LBMParams& params, 
                 const velocity_array_t v,
                 const weights_t t,
                 const real_t* rho,
                 const real_t* ux,
                 const real_t* uy,
                 real_t* feq)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t usqr = 3.0 / 2 * (ux[index] * ux[index] +
                               uy[index] * uy[index]);
      
      for (int ipop = 0; ipop < npop; ++ipop) {
        real_t cu = 3 * (v(ipop,0) * ux[index] + 
                         v(ipop,1) * uy[index]);

        int index_f = index + ipop * nx * ny;
        feq[index_f] = rho[index] * t(ipop) * (1 + cu + 0.5*cu*cu - usqr);
      }
      
    } // end for i
  } // end for j

} // equilibrium

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams& params, int* obstacle)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t x = 1.0*i;
      real_t y = 1.0*j;

      obstacle[index] = (x-cx)*(x-cx) + (y-cy)*(y-cy) < r*r ? 1 : 0;

    } // end for i
  } // end for j

} // init_obstacle_mask

// ======================================================
// ======================================================
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1-dir) * uLB * (1 + 1e-4 * sin(j/ly*2*M_PI));

} // compute_vel

// ======================================================
// ======================================================
void initialize_macroscopic_variables(const LBMParams& params, 
                                      real_t* rho,
                                      real_t* ux,
                                      real_t* uy)
{

  const int nx = params.nx;
  const int ny = params.ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      rho[index] = 1.0;
      ux[index]  = compute_vel(0, i, j, params.uLB, params.ly);
      uy[index]  = compute_vel(1, i, j, params.uLB, params.ly);

    } // end for i
  } // end for j

} // initialize_macroscopic_variables

// ======================================================
// ======================================================
void border_outflow(const LBMParams& params, real_t* fin)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx*ny;

  const int i1 = nx-1;
  const int i2 = nx-2;

  for (int j=0; j<ny; ++j) {

    int index1 = i1 + nx * j;
    int index2 = i2 + nx * j;

    fin[index1 + 6*nxny] = fin[index2 + 6*nxny];
    fin[index1 + 7*nxny] = fin[index2 + 7*nxny];
    fin[index1 + 8*nxny] = fin[index2 + 8*nxny];

  } // end for j

} // border_outflow

// ======================================================
// ======================================================
void border_inflow(const LBMParams& params, const real_t* fin, 
                   real_t* rho, real_t* ux, real_t* uy)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx*ny;

  const int i = 0;

  for (int j = 0; j < ny; ++j) {

    int index = i + nx * j;

    ux[index] = compute_vel(0, i, j, params.uLB, params.ly);
    uy[index] = compute_vel(1, i, j, params.uLB, params.ly);
    rho[index] = 1/(1-ux[index]) * 
      (    fin[index+3*nxny] + fin[index+4*nxny] + fin[index+5*nxny] +
        2*(fin[index+6*nxny] + fin[index+7*nxny] + fin[index+8*nxny]) );

  } // end for j

} // border_inflow

// ======================================================
// ======================================================
void update_fin_inflow(const LBMParams& params, const real_t* feq, 
                       real_t* fin)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx*ny;

  const int i = 0;

  for (int j = 0; j < ny; ++j) {

    int index = i + nx * j;
    
    //fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]

    fin[index+0*nxny] = feq[index+0*nxny] + fin[index+8*nxny] - feq[index+8*nxny];
    fin[index+1*nxny] = feq[index+1*nxny] + fin[index+7*nxny] - feq[index+7*nxny];
    fin[index+2*nxny] = feq[index+2*nxny] + fin[index+6*nxny] - feq[index+6*nxny];

  } // end for j

} // update_fin_inflow
  
// ======================================================
// ======================================================
void compute_collision(const LBMParams& params, 
                       const real_t* fin,
                       const real_t* feq,
                       real_t* fout)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const int nxny = nx*ny;

  const int npop = LBMParams::npop;
  const real_t omega = params.omega;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      for (int ipop=0; ipop<npop; ++ipop) {

        int index_f = index + ipop*nxny;

        fout[index_f] = fin[index_f] - omega * (fin[index_f]-feq[index_f]);

      } // end for ipop

    } // end for i
  } // end for j

} // compute_collision

// ======================================================
// ======================================================
void update_obstacle(const LBMParams &params, 
                     const real_t* fin,
                     const int* obstacle, 
                     real_t* fout)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nxny = nx*ny;
  const int npop = LBMParams::npop;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      if (obstacle[index]==1) {

        for (int ipop = 0; ipop < npop; ++ipop) {

          int index_out = index +    ipop  * nxny;
          int index_in  = index + (8-ipop) * nxny;

          fout[index_out] = fin[index_in];

        } // end for ipop

      } // end inside obstacle

    } // end for i
  } // end for j

} // update_obstacle

// ======================================================
// ======================================================
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout,
               real_t* fin)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int nxny = nx*ny;
  const int npop = LBMParams::npop;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      for (int ipop = 0; ipop < npop; ++ipop) {

        int index_in = index + ipop * nxny;

        int i_out = i-v(ipop,0);
        if (i_out<0)
          i_out += nx;
        if (i_out>nx-1)
          i_out -= nx;

        int j_out = j-v(ipop,1);
        if (j_out<0)
          j_out += ny;
        if (j_out>ny-1)
          j_out -= ny;

        int index_out = i_out + nx*j_out + ipop*nxny;

        fin[index_in] = fout[index_out];

      } // end for ipop

    } // end for i

  } // end for j

} // streaming
