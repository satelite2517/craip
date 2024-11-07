cmake_minimum_required(VERSION 3.0.2)
project(custom_msgs)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  ContourLabel.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  custom_msgs
)

catkin_package(
 CATKIN_DEPENDS rospy std_msgs message_runtime geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

