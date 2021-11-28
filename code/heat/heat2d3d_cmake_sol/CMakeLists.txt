# require cmake 3.18 for modern cmake cuda support
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

message("Using CMake version ${CMAKE_VERSION}")

# CMP0104 is about CUDA_ARCHITECTURES
# see https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("Using cmake policy CMP0104 OLD")
    cmake_policy(SET CMP0104 OLD)
else()
    message("Using cmake policy CMP0104 NEW")
    cmake_policy(SET CMP0104 NEW)
endif()

project(heat_solver LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# Prevent from build in source tree
include(preventBuildInSource)

# Init build type: Release, Debug, ...
include(initBuildType)

# useful for debugging cmake
include(print_target_properties)

# you can disable CUDA, and only build the CPU version
option(USE_CUDA "Use CUDA" ON)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------------------#
# Cuda configuration
#------------------------------------------------------------------#

if (USE_CUDA)

  enable_language("CUDA")

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fno-strict-aliasing -lineinfo")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${cuda_tutorial_SOURCE_DIR}/cuda_helper")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_35,code=sm_35")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_37,code=sm_37")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")

  # we also need extra stuff to make sure compile flags are correctly
  # passed to nvcc / host compiler
  include(protect_nvcc_flags)
  include(protect_pthread_flag)

endif(USE_CUDA)

# other third party libraries
find_package(MathGL2)

# make the include path visible to all targets
include_directories(${CMAKE_SOURCE_DIR}/common)

find_package(Threads)
find_package(GLUT)

find_package(OpenMP)

####################################
# Some build options
####################################
# mathGL library
option (MATHGL_OUTPUT_ENABLED "Enable output file format to be MGL (http://mathgl.sourceforge.net/doc_en/Main.html)" OFF)
if (MATHGL_OUTPUT_ENABLED)
  add_definitions (-DUSE_MGL)
  message (STATUS "MathGL output enabled")
endif()

# double precision
option (DOUBLE_PRECISION_ENABLED "Enable double precision computation" OFF)
if (DOUBLE_PRECISION_ENABLED)
  add_definitions (-DUSE_DOUBLE)
  message (STATUS "Double precision computations enabled")
endif()

option (DEVICE_SYNC_FORCED "enforce device sync after each kernel call" OFF)
if (DEVICE_SYNC_FORCED)
  add_definitions (-DALWAYS_SYNC_GPU)
endif()

##################
# HOST/CPU BUILD
##################

add_subdirectory(common)

# CPU executable
add_executable(heat_solver_cpu "")
target_sources(heat_solver_cpu PUBLIC heat_solver_cpu.cpp)
target_link_libraries(heat_solver_cpu PUBLIC heat::heatCommon)
if (OpenMP_FOUND)
  target_link_libraries(heat_solver_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()
if (MATHGL_OUTPUT_ENABLED)
  target_link_libraries(heat_solver_cpu PUBLIC mgl )
endif()

#############
# GPU BUILD
#############

if(USE_CUDA)

  ################################
  # list of 2D executables for GPU
  ################################

  # heat2d_solver_gpu_naive
  add_executable ( heat2d_solver_gpu_naive
    heat2d_solver_gpu_naive.cu
    )
  target_include_directories( heat2d_solver_gpu_naive PUBLIC cuda_helper)
  target_link_libraries( heat2d_solver_gpu_naive
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat2d_solver_gpu_naive
      mgl
      )
  endif()

  # heat2d_solver_gpu_naive_managed
  add_executable ( heat2d_solver_gpu_naive_managed
    heat2d_solver_gpu_naive_managed.cu
    )
  target_include_directories( heat2d_solver_gpu_naive_managed PUBLIC cuda_helper)
  target_link_libraries( heat2d_solver_gpu_naive_managed
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat2d_solver_gpu_naive_managed
      mgl
      )
  endif()

  # heat2d_solver_gpu_shmem1
  add_executable ( heat2d_solver_gpu_shmem1
    heat2d_solver_gpu_shmem1.cu
    )
  target_include_directories( heat2d_solver_gpu_shmem1 PUBLIC cuda_helper)
  target_link_libraries( heat2d_solver_gpu_shmem1
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat2d_solver_gpu_shmem1
      mgl
      )
  endif()

  # heat2d_solver_gpu_shmem2
  add_executable ( heat2d_solver_gpu_shmem2
    heat2d_solver_gpu_shmem2.cu
    )
  target_include_directories( heat2d_solver_gpu_shmem2 PUBLIC cuda_helper)
  target_link_libraries( heat2d_solver_gpu_shmem2
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat2d_solver_gpu_shmem2
      mgl
      )
  endif()

  ################################
  # list of 3D executables for GPU
  ################################

  # heat3d_solver_gpu_naive
  add_executable ( heat3d_solver_gpu_naive
    heat3d_solver_gpu_naive.cu
    )
  target_include_directories( heat3d_solver_gpu_naive PUBLIC cuda_helper)
  target_link_libraries( heat3d_solver_gpu_naive
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat3d_solver_gpu_naive
      mgl
      )
  endif()

  # heat3d_solver_gpu_shmem1
  add_executable ( heat3d_solver_gpu_shmem1
    heat3d_solver_gpu_shmem1.cu
    )
  target_include_directories( heat3d_solver_gpu_shmem1 PUBLIC cuda_helper)
  target_link_libraries( heat3d_solver_gpu_shmem1
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat3d_solver_gpu_shmem1
      mgl
      )
  endif()

  # heat3d_solver_gpu_shmem2
  add_executable ( heat3d_solver_gpu_shmem2
    heat3d_solver_gpu_shmem2.cu
    )
  target_include_directories( heat3d_solver_gpu_shmem2 PUBLIC cuda_helper)
  target_link_libraries( heat3d_solver_gpu_shmem2
    HeatCommon
    )
  if (MATHGL_OUTPUT_ENABLED)
    target_link_libraries(heat3d_solver_gpu_shmem2
      mgl
      )
  endif()

  ##########################
  # CPU BUILD with GLUT
  ##########################
  # set ( heat2d_glut_cpu_SRCS
  #   main_glut.cpp
  #   )
  # add_executable (heat2d_glut_cpu ${heat2d_glut_cpu_SRCS})
  # set_target_properties (heat2d_glut_cpu PROPERTIES COMPILE_FLAGS
  #   -pthread)
  # target_link_libraries (heat2d_glut_cpu
  #   HeatCommon
  #   pthread glut GL GLEW
  #   )

  ##########################
  # GPU BUILD with GLUT
  ##########################
  # set ( heat2d_glut_gpu_SRCS
  #   main_glut_gpu.cu
  #   )
  # add_executable (heat2d_glut_gpu ${heat2d_glut_gpu_SRCS})
  # set_target_properties (heat2d_glut_gpu PROPERTIES COMPILE_FLAGS
  #   -pthread)
  # target_link_libraries (heat2d_glut_gpu
  #   HeatCommon
  #   pthread glut GL GLEW
  #   )

endif(USE_CUDA)

# copy the default parameter file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/heatEqSolver.par"
  "${CMAKE_CURRENT_BINARY_DIR}/heatEqSolver.par")

# add_custom_target( FinalMessage ALL
#     ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Compilation is over!"
#     ${CMAKE_COMMAND} -E message(STATUS "USE_CUDA: ${USE_CUDA}")
#     COMMENT "Final Message" )
# add_dependencies( FinalMessage ${ALL_MY_TARGETS} )


#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")

message("  CMake version          : ${CMAKE_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()

message("  C++ Compiler Id        : ${CMAKE_CXX_COMPILER_ID}")
message("  C++ Compiler version   : ${CMAKE_CXX_COMPILER_VERSION}")
message("  C++ Compiler wrapper   : ${CMAKE_CXX_COMPILER_WRAPPER}")
message("  C++ Compiler           :  ${CMAKE_CXX_COMPILER}")

message("  USE_CUDA               : ${USE_CUDA}")
if (USE_CUDA)
  message("  CUDA compiler ID       : ${CMAKE_CUDA_COMPILER_ID}")
  message("  CUDA compiler Version  : ${CMAKE_CUDA_COMPILER_VERSION}")
  message("  CUDA Compiler          : ${CMAKE_CUDA_COMPILER}")
  message("  CUDA Compiler exec     : ${CUDA_NVCC_EXECUTABLE}")
  message("  CUDA Compile flags     : ${CMAKE_CUDA_FLAGS}")
  message("  CUDA toolkit inc       : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif (USE_CUDA)

message("")

if (OpenMP_FOUND)
  message("  OpenMP found")
  message("  OpenMP CXX flags       : ${OpenMP_CXX_FLAGS}")
else()
  message("  OpenMP not found")
endif()
