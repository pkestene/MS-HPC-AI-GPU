add_subdirectory(utils)
add_subdirectory(lbm)

add_executable(lbmFlowAroundCylinder "")

target_sources(lbmFlowAroundCylinder PRIVATE
  main.cpp)

# TODO : uncomment when build OpenACC version
#target_compile_options(lbmFlowAroundCylinder
#  PUBLIC
#  ${MY_OpenACC_CXX_FLAGS})

target_include_directories(lbmFlowAroundCylinder
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)

# TODO : uncomment when build OpenACC version
#target_link_options(lbmFlowAroundCylinder 
#  PUBLIC
#  ${OpenACC_CXX_FLAGS})

target_link_libraries(lbmFlowAroundCylinder
  PUBLIC
  lbm::core
  lbm::config
  lbm::lodepng
  lbm::vtk)

configure_file(flowAroundCylinder.ini flowAroundCylinder.ini COPYONLY)
