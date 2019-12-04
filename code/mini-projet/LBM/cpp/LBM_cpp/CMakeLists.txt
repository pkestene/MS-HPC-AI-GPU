cmake_minimum_required(VERSION 3.10)
project(cuda-proj-tmpl LANGUAGES CXX C)

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

# a simple macro to build executable avoiding copy paste
#include(build_macro)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

#####################################################################
include(CheckLanguage)

# the following will populate or not variable CMAKE_CUDA_COMPILER
check_language(CUDA)

if (NOT ${CMAKE_CUDA_COMPILER} STREQUAL "CMAKE_CUDA_COMPILER-NOTFOUND")

  enable_language(CUDA)

  #find_package(CUDA REQUIRED)
  find_package(CUDALibs REQUIRED COMPONENTS CUBLAS )

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

else()

  message("CMAKE_CUDA_COMPILER is not set !")

endif()

# other third party libraries
find_package(OpenMP)
#find_package(OpenGL)
#find_package(GLUT)

# we also need extra stuff to make sure compile flags are correctly
# passed to nvcc / host compiler
include(protect_nvcc_flags)
include(protect_pthread_flag)

option (USE_DOUBLE "build with double precision" OFF)
if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif(USE_DOUBLE)

##########################################################
##########################################################
##########################################################
##########################################################

# build some executable
add_subdirectory(src)
add_subdirectory(test)

# Testing
#enable_testing()

#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} " 
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("")
message("  OpenMP compile flags : ${OpenMP_CXX_FLAGS}")
message("")
#message(" Thrid party libraries")
#message("  OpenGL found : ${OPENGL_FOUND}")
