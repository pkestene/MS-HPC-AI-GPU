#include "CpuTimerOmp.h"

#include <stdlib.h>
#include <stdio.h>

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// CpuTimerOmp class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
CpuTimerOmp::CpuTimerOmp() {
  start_time = 0.0;

  start();
} // CpuTimerOmp::CpuTimerOmp

// =======================================================
// =======================================================
CpuTimerOmp::CpuTimerOmp(double t) 
{
  
  start_time = 0.0;
  total_time = t;
  
} // CpuTimerOmp::CpuTimerOmp

// =======================================================
// =======================================================
CpuTimerOmp::CpuTimerOmp(CpuTimerOmp const& aCpuTimerOmp) : 
  start_time(aCpuTimerOmp.start_time), 
  total_time(aCpuTimerOmp.total_time)
{
} // CpuTimerOmp::CpuTimerOmp

// =======================================================
// =======================================================
CpuTimerOmp::~CpuTimerOmp()
{
} // CpuTimerOmp::~CpuTimerOmp

// =======================================================
// =======================================================
void CpuTimerOmp::start() 
{
  
  start_time = omp_get_wtime();
  
} // CpuTimerOmp::start

// =======================================================
// =======================================================
void CpuTimerOmp::stop()
{
  double now = omp_get_wtime();
  double accum = now - start_time;
  
  total_time += accum*1e3; // in milliseconds
  
} // CpuTimerOmp::stop

// =======================================================
// =======================================================
double CpuTimerOmp::elapsed() const
{
  
  return total_time;  // in milliseconds
  
} // CpuTimerOmp::elapsed
