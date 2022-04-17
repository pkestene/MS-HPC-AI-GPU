#ifndef LBM_FLOW_UTILS_KERNELS_H
#define LBM_FLOW_UTILS_KERNELS_H

// ================================================================
// ================================================================
__global__ 
void macroscopic_kernel(const LBMParams params,
                        const velocity_array_t v,
                        const real_t* fin_d,
                        real_t* rho_d,
                        real_t* ux_d,
                        real_t* uy_d)
{

  // TODO

} // macroscopic_kernel

// ================================================================
// ================================================================
__global__ void equilibrium_kernel(const LBMParams params,
                                   const velocity_array_t v,
                                   const weights_t t,
                                   const real_t *rho_d, 
                                   const real_t *ux_d,
                                   const real_t *uy_d, 
                                   real_t *feq_d)
{

  // TODO

} // equilibrium_kernel

// ================================================================
// ================================================================
__global__ void border_outflow_kernel(const LBMParams params, 
                                      real_t *fin_d)
{

  // TODO

} // border_outflow_kernel

// ================================================================
// ================================================================
__global__ void border_inflow_kernel(const LBMParams params, 
                                     const real_t *fin_d,
                                     real_t *rho_d,
                                     real_t *ux_d,
                                     real_t *uy_d)
{

  // TODO

} // border_inflow_kernel

// ================================================================
// ================================================================
__global__ void update_fin_inflow_kernel(const LBMParams params, 
                                         const real_t *feq_d,
                                         real_t *fin_d)
{

  // TODO

} // update_fin_inflow_kernel

// ================================================================
// ================================================================
__global__ void compute_collision_kernel(const LBMParams params,
                                         const real_t *fin_d, 
                                         const real_t *feq_d, 
                                         real_t *fout_d)
{

  // TODO

} // compute_collision_kernel

// ================================================================
// ================================================================
__global__ void update_obstacle_kernel(const LBMParams params,
                                       const real_t *fin_d, 
                                       const int *obstacle_d, 
                                       real_t *fout_d)
{

  // TODO

} // update_obstacle_kernel

// ================================================================
// ================================================================
__global__ void streaming_kernel(const LBMParams params,
                                 const velocity_array_t v,
                                 const real_t *fout_d,
                                 real_t *fin_d)
{

  // TODO

} // streaming_kernel

#endif // LBM_FLOW_UTILS_KERNELS_H
