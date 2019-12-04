#ifndef HEAT3D_KERNEL_CPU_H_
#define HEAT3D_KERNEL_CPU_H_

void heat3d_ftcs_cpu_order2( float* data, float* dataNext );
void heat3d_ftcs_cpu_order4( float* data, float* dataNext );

#endif // HEAT3D_KERNEL_CPU_H_
