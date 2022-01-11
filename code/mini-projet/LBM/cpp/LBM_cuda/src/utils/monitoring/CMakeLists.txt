add_library(lbmMonitoring "")
target_sources(lbmMonitoring PRIVATE
  OpenMPTimer.cpp
  SimpleTimer.cpp)

if (CMAKE_CUDA_COMPILER_LOADED)
target_sources(lbmMonitoring PRIVATE
  my_cuda_utils.cu)
endif(CMAKE_CUDA_COMPILER_LOADED)


target_include_directories(lbmMonitoring
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

#
# When compiling with cuda enabled,just uncomment
# the COMPILE_LANGUAGE:CUDA line below
#
target_compile_options(lbmMonitoring PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
  #$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${OpenMP_CXX_FLAGS}>
  )

if(OpenMP_CXX_FOUND)
  target_link_libraries(lbmMonitoring PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_CXX_FOUND)

add_library(lbm::common ALIAS lbmMonitoring)
