cmake_minimum_required(VERSION 2.8.3)
project(tams_biotac)

set(HEADERS
  include/${PROJECT_NAME}/subtract_temperature.h
  include/${PROJECT_NAME}/filter.h
  include/${PROJECT_NAME}/normalization.h
  include/${PROJECT_NAME}/contacts.h
  include/${PROJECT_NAME}/visualize_electrodes.h
  include/${PROJECT_NAME}/tactile_calibrated.h
  include/${PROJECT_NAME}/force.h
)

set(SOURCE_FILES
  src/subtract_temperature.cpp
  src/filter.cpp
  src/normalization.cpp
  src/contacts.cpp
  src/visualize_electrodes.cpp
  src/tactile_calibrated.cpp
  src/force.cpp
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  std_srvs
  geometry_msgs
  visualization_msgs
  message_generation
  sr_robot_msgs
  sr_hardware_interface
)

# Support new yaml-cpp API.
find_package(PkgConfig)
pkg_check_modules(NEW_YAMLCPP yaml-cpp>=0.5)
if(NEW_YAMLCPP_FOUND)
  add_definitions(-DHAVE_NEW_YAMLCPP)
endif(NEW_YAMLCPP_FOUND)

find_library(YAML yaml-cpp REQUIRED)

add_message_files(
  FILES
  Contact.msg
  ContactArray.msg
  Biotac.msg
  BiotacAll.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib ${SOURCE_FILES} ${HEADERS})
add_dependencies(${PROJECT_NAME}_lib ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_lib ${catkin_LIBRARIES} ${YAML})

add_executable(tams_biotac_node src/tams_biotac_node.cpp)
add_dependencies(tams_biotac_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(tams_biotac_node ${catkin_LIBRARIES} ${PROJECT_NAME}_lib)

add_executable(tams_biotac_calibrated_node src/tams_biotac_calibrated_node.cpp)
add_dependencies(tams_biotac_calibrated_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(tams_biotac_calibrated_node ${catkin_LIBRARIES} ${PROJECT_NAME}_lib)

add_executable(temperature src/temperature.cpp)
target_link_libraries(temperature ${catkin_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
 DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
 FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
