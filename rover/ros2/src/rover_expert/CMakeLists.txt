cmake_minimum_required(VERSION 3.8)
project(rover_expert)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/expert_driver.py
    scripts/episode_recorder.py
    scripts/run_episode.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
