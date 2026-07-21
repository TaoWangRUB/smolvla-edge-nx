cmake_minimum_required(VERSION 3.8)
project(rover_runtime)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/tracker_node.py
    scripts/chunk_client_node.py
    scripts/goal_memory_node.py
    scripts/goal_projection.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
