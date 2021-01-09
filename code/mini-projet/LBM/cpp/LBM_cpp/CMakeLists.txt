cmake_minimum_required(VERSION 3.10)
project(LBMcpp LANGUAGES CXX C)

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

# other third party libraries
find_package(OpenMP)
#find_package(OpenACC)


option (USE_DOUBLE "build with double precision" OFF)
if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif(USE_DOUBLE)

# TODO : uncomment the following for OpenAcc
# # set OpenACC link flags
# set(OpenACC_CXX_LINK ${OpenACC_CXX_FLAGS})
# 
# # build OpenACC compile flags
# set(MY_OpenACC_CXX_FLAGS "")
# set( value_list ${OpenACC_TARGET_FLAGS} )
# separate_arguments(value_list)
# foreach (item ${value_list})
#   list(APPEND MY_OpenACC_CXX_FLAGS ${item})
# endforeach()


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
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} " 
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("")
# message("  OpenACC spec data     : ${OpenACC_CXX_SPEC_DATE}")
# message("  OpenACC compile flags : ${OpenACC_CXX_FLAGS}")
# # get openacc target flags from command line
# # example : cmake -DOpenACC_TARGET_FLAGS:STRING="-ta=tesla:cc80 -Minfo=all"
#message("  OpenACC target flags  : ${OpenACC_TARGET_FLAGS}")
#message("  OpenACC complete compile flags : ${MY_OpenACC_CXX_FLAGS}")
message("")
#message(" Thrid party libraries")
#message("  OpenGL found : ${OPENGL_FOUND}")
