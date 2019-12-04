#ifndef __CPU_TIMER_H__
#define __CPU_TIMER_H__

#include <time.h>
#include <sys/time.h> // for gettimeofday and struct timeval

typedef struct timeval timeval_t;

/**
 * \brief a simple CpuTimer class.
 */
class CpuTimer
{
public:
  /** default constructor, timing starts rightaway */
  CpuTimer();
  
  CpuTimer(double t);
  CpuTimer(CpuTimer const& aCpuTimer);
  ~CpuTimer();
  
  /** start time measure */
  void start();
  
  /** stop time measure and add result to total_time */
  void stop();
  
  /** return elapsed time in seconds (as stored in total_time) */
  double elapsed() const;
  
protected:
  timeval_t start_time;
  
  /** store total accumulated timings */
  double    total_time;  // in milli-seconds
  
}; // class CpuTimer

#endif // __CPU_TIMER_H__
