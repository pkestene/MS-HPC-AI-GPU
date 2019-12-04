#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer
{
      cudaEvent_t start_time;
      cudaEvent_t stop_time;
 
      GpuTimer()
      {
            cudaEventCreate(&start_time);
            cudaEventCreate(&stop_time);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start_time);
            cudaEventDestroy(stop_time);
      }
 
      void start()
      {
            cudaEventRecord(start_time, 0);
      }
 
      void stop()
      {
	    cudaEventSynchronize(stop_time);
	    cudaEventRecord(stop_time, 0);
	    cudaEventSynchronize(stop_time);
      }
 
      float elapsed()
      {
            float elapsed;
            
            cudaEventElapsedTime(&elapsed, start_time, stop_time);
            return elapsed;
      }
};

#endif  /* __GPU_TIMER_H__ */
