cmake_minimum_required(VERSION 3.8)
project(rover_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf config worlds launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/scene_manager.py
    scripts/state_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
