add_library(lbmCore "")
target_sources(lbmCore PRIVATE
  LBMParams.cpp
  lbmFlowUtils.cpp
  LBMSolver.cpp)

target_include_directories(lbmCore
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils)

add_library(lbm::core ALIAS lbmCore)
