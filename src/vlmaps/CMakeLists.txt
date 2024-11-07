cmake_minimum_required(VERSION 3.0.2)
project(vlmaps)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    custom_msgs
    nav_msgs
    visualization_msgs
    message_filters
)

catkin_package(
    LIBRARIES vlmaps
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/map.py
  scripts/pointcloud.py
  scripts/scf_wrapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)