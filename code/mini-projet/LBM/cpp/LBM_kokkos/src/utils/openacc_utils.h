#ifndef OPENACC_UTILS_H
#define OPENACC_UTILS_H

#include <iostream> // std::cout
#include <cstdlib> // for std::getenv

#ifdef _OPENACC
#include <openacc.h>
#endif // _OPENACC

// =========================================================
// =========================================================
void print_openacc_version()
{

#ifdef _OPENACC
  printf("OpenACC version %d\n",_OPENACC);
#else
  printf("OpenACC is not enabled\n");
#endif // _OPENACC

} // print_openacc_version

// =========================================================
// =========================================================
void init_openacc() 
{

#ifdef _OPENACC
  
  // try to find a gpu device
  int ngpus = acc_get_num_devices( acc_device_nvidia );
  
  if( ngpus ){
    // use the first one
    int gpunum = 0;
    acc_set_device_num( gpunum, acc_device_nvidia );
  } else {
    /* no NVIDIA GPUs available */ 
    acc_set_device_type( acc_device_host); 
  }

  // It's recommended to perform one-off initialization
  //  of GPU-devices before any OpenACC regions
  acc_init(acc_get_device_type()) ;

#endif // _OPENACC

} // init_openacc

#endif // OPENACC_UTILS_H
