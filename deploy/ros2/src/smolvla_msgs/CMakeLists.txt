cmake_minimum_required(VERSION 3.8)
project(smolvla_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SimObservation.msg"
  "msg/TickEvent.msg"
  "msg/PolicyRequest.msg"
  "msg/PolicyChunk.msg"
  DEPENDENCIES sensor_msgs std_msgs
)

ament_package()
