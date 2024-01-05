#
# ConfigMap class test (src/util/config)
#

##############################################
add_executable(ConfigMapTest
  ConfigMapTest.cpp)

target_link_libraries(ConfigMapTest
  PRIVATE
  lbm::config)
