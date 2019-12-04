#include "CpuTimer.h"

#include <stdlib.h>
#include <stdio.h>

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// CpuTimer class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
CpuTimer::CpuTimer() {
  start_time.tv_sec = 0;
  start_time.tv_usec = 0;
  total_time = 0.0;
  start();
} // CpuTimer::CpuTimer

  // =======================================================
  // =======================================================
CpuTimer::CpuTimer(double t) 
{
  
  //start_time.tv_sec = time_t(t);
  //start_time.tv_usec = (t - start_time.tv_sec) * 1e6;
  start_time.tv_sec = 0;
  start_time.tv_usec = 0;
  total_time = t;
  
} // CpuTimer::CpuTimer

  // =======================================================
  // =======================================================
CpuTimer::CpuTimer(CpuTimer const& aCpuTimer) : start_time(aCpuTimer.start_time), total_time(aCpuTimer.total_time)
{
} // CpuTimer::CpuTimer

  // =======================================================
  // =======================================================
CpuTimer::~CpuTimer()
{
} // CpuTimer::~CpuTimer

  // =======================================================
  // =======================================================
void CpuTimer::start() 
{
  
  if (-1 == gettimeofday(&start_time, 0))
    throw std::runtime_error("CpuTimer: Couldn't initialize start_time time");
  
} // CpuTimer::start

  // =======================================================
  // =======================================================
void CpuTimer::stop()
{
  double accum;
  timeval now;
  if (-1 == gettimeofday(&now, 0))
    throw std::runtime_error("Couldn't get current time");

  if (now.tv_sec == start_time.tv_sec)
    accum = double(now.tv_usec - start_time.tv_usec) * 1e-6;
  else
    accum = double(now.tv_sec - start_time.tv_sec) + 
      (double(now.tv_usec - start_time.tv_usec) * 1e-6);
  
  total_time += accum*1e-3; // in milliseconds
  
} // CpuTimer::stop

// =======================================================
// =======================================================
double CpuTimer::elapsed() const
{
  
  return total_time;
  
} // CpuTimer::elapsed
