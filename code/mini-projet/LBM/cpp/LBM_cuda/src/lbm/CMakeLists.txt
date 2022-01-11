add_library(lbmCore "")
target_sources(lbmCore PRIVATE
  LBMParams.cpp
  lbmFlowUtils.cu
  LBMSolver.cu)

target_include_directories(lbmCore
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils)

add_library(lbm::core ALIAS lbmCore)
