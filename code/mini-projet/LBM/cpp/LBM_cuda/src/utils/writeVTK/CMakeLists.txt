add_library(saveVTK "")
target_sources(saveVTK PRIVATE
  saveVTK.cpp)

target_include_directories(saveVTK
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )

add_library(lbm::vtk ALIAS saveVTK)

# build an example
add_executable(testVTK "")
target_sources(testVTK PRIVATE
  testVTK.cpp)
target_link_libraries(testVTK PRIVATE
  lbm::vtk
  lbm::config
  lbm::core)
