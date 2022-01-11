add_library(config "")

target_sources(config
  PRIVATE
  ConfigMap.cpp
  inih/ini.cpp
  inih/INIReader.cpp
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils
  ) 

#Add an alias so that library can be used inside the build tree
add_library(lbm::config ALIAS config)
