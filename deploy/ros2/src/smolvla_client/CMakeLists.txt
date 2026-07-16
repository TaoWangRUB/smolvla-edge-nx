cmake_minimum_required(VERSION 3.8)
project(smolvla_client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(smolvla_msgs REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP REQUIRED grpc++)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# C++ stubs from the SAME proto the Python stack uses (single source of truth, task 3.3).
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../../client_server/proto/policy.proto)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto_gen)
file(MAKE_DIRECTORY ${GEN_DIR})
add_custom_command(
  OUTPUT ${GEN_DIR}/policy.pb.cc ${GEN_DIR}/policy.pb.h
         ${GEN_DIR}/policy.grpc.pb.cc ${GEN_DIR}/policy.grpc.pb.h
  COMMAND protobuf::protoc
  ARGS --cpp_out=${GEN_DIR} --grpc_out=${GEN_DIR}
       --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
       -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../client_server/proto
       ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Generating C++ gRPC stubs from policy.proto"
)

add_library(policy_proto STATIC
  ${GEN_DIR}/policy.pb.cc
  ${GEN_DIR}/policy.grpc.pb.cc
)
target_include_directories(policy_proto PUBLIC ${GEN_DIR} ${GRPCPP_INCLUDE_DIRS})
target_link_libraries(policy_proto protobuf::libprotobuf ${GRPCPP_LIBRARIES})
set_property(TARGET policy_proto PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(async_client src/async_client.cpp)
target_include_directories(async_client PRIVATE include)
target_link_libraries(async_client policy_proto)
ament_target_dependencies(async_client rclcpp std_msgs sensor_msgs smolvla_msgs)

install(TARGETS async_client DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_aggregate test/test_aggregate.cpp)
  target_include_directories(test_aggregate PRIVATE include)
  target_compile_definitions(test_aggregate PRIVATE
    FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/aggregate_fixtures.txt")
endif()

ament_package()
