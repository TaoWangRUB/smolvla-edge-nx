# C++ ONNX Runtime inference server (Stage 2b, cpp-inference-server / design D6).
# Builds a gRPC Policy server backed by ONNX Runtime (CUDA/TensorRT EP) over the SAME
# policy.proto the Python server and ROS2 client use — so clients run unmodified (task 5.4).
cmake_minimum_required(VERSION 3.16)
project(smolvla_cpp_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# gRPC C++ plugin + runtime: prefer CONFIG package, fall back to pkg-config (Debian layout)
if(NOT gRPC_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED grpc++)
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  set(GRPCPP_LIBS ${GRPCPP_LIBRARIES})
else()
  set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  set(GRPCPP_LIBS gRPC::grpc++)
endif()

# ONNX Runtime (from the GPU C++ release unpacked to $ORT_ROOT in the image)
set(ORT_ROOT "$ENV{ORT_ROOT}" CACHE PATH "ONNX Runtime C++ release root")
find_library(ORT_LIB onnxruntime HINTS ${ORT_ROOT}/lib REQUIRED)

# --- generate proto + grpc stubs from the shared policy.proto -------------------------------
set(PROTO ${CMAKE_CURRENT_SOURCE_DIR}/../client_server/proto/policy.proto)
get_filename_component(PROTO_DIR ${PROTO} DIRECTORY)
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN})
set(PB ${GEN}/policy.pb.cc)
set(GRPC_PB ${GEN}/policy.grpc.pb.cc)
add_custom_command(
  OUTPUT ${PB} ${GRPC_PB}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --experimental_allow_proto3_optional
          --proto_path=${PROTO_DIR} --cpp_out=${GEN} --grpc_out=${GEN}
          --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} ${PROTO}
  DEPENDS ${PROTO}
  COMMENT "protoc: policy.proto -> C++ + gRPC stubs")

add_executable(policy_server src/server.cpp ${PB} ${GRPC_PB})
target_include_directories(policy_server PRIVATE ${GEN} ${ORT_ROOT}/include
  ${Protobuf_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GRPCPP_INCLUDE_DIRS})
target_link_libraries(policy_server PRIVATE
  ${ORT_LIB} ${GRPCPP_LIBS} protobuf::libprotobuf ${OpenCV_LIBS} Threads::Threads)
