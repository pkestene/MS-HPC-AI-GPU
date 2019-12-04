#ifndef __CPU_TIMER_OMP_H__
#define __CPU_TIMER_OMP_H__

#include <omp.h>

/**
 * \brief a simple CpuTimerOmp class.
 */
class CpuTimerOmp
{
public:
  /** default constructor, timing starts rightaway */
  CpuTimerOmp();
  
  CpuTimerOmp(double t);
  CpuTimerOmp(CpuTimerOmp const& aCpuTimerOmp);
  ~CpuTimerOmp();
  
  /** start time measure */
  void start();
  
  /** stop time measure and add result to total_time */
  void stop();
  
  /** return elapsed time in seconds (as stored in total_time) */
  double elapsed() const;
  
protected:
  double    start_time;
  
  /** store total accumulated timings */
  double    total_time;  // in milli-seconds
  
}; // class CpuTimerOmp

#endif // __CPU_TIMER_OMP_H__
