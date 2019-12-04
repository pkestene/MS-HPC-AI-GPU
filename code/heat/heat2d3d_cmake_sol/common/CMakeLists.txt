
# libHeatCommon source files list
add_library(HeatCommon "")

target_sources(HeatCommon 
  PRIVATE
  output.cpp 
  param.cpp 
  misc.cpp
  Timer.cpp
  heat_kernel_cpu.cpp)

# the following is necessary as every target that will link
# with HeatCommon will inherit compile and target flags.
# If the target executable is build with nvcc, that the flags
# specified here must take this into account.
# When using nvcc, the OpenMP compile flags must be protected
# by -Xcompiler (to be passed only the host compiler, not device)
target_compile_options(HeatCommon PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if (OpenMP_FOUND)
  target_link_libraries(HeatCommon PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

add_library(heat::heatCommon ALIAS HeatCommon)

