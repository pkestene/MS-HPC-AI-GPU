add_subdirectory(utils)
add_subdirectory(lbm)

add_executable(lbmFlowAroundCylinder "")
target_sources(lbmFlowAroundCylinder PRIVATE
  main.cu)
target_include_directories(lbmFlowAroundCylinder
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lbmFlowAroundCylinder
  PUBLIC
  lbm::core
  lbm::config
  lbm::lodepng
  lbm::vtk)

configure_file(flowAroundCylinder.ini flowAroundCylinder.ini COPYONLY)
