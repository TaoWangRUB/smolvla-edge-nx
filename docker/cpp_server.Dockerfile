# smolvla-edge C++ ONNX Runtime inference server (Stage 2b, cpp-inference-server / design D6).
#
# FROM the sim image because it already carries the CUDA 12.4 + cuDNN 9 + cuBLAS math stack
# (pytorch base's pip nvidia wheels), which ONNX Runtime's CUDA EP needs — the alternative,
# adding CUDA to the Jazzy ros2 image, is heavier. This layer adds only the C++ toolchain the
# server needs: gRPC/protobuf C++, OpenCV (serving-side resize_with_pad), and the ONNX Runtime
# GPU C++ release (the pip wheel ships no C++ headers).
#
# The nvidia wheel lib dirs are put on LD_LIBRARY_PATH (the rover's pattern) so both the ORT CUDA
# provider and the CUDA runtime resolve at load time — setting it inside the process is too late.

FROM smolvla-edge:sim

ENV DEBIAN_FRONTEND=noninteractive
ARG ORT_VERSION=1.20.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates cmake build-essential pkg-config \
    libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime GPU C++ release (headers + libonnxruntime.so + CUDA/TensorRT provider .so)
RUN cd /opt && \
    curl -fsSL -o ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" && \
    tar xzf ort.tgz && mv onnxruntime-linux-x64-gpu-${ORT_VERSION} /opt/onnxruntime && rm ort.tgz

# TensorRT runtime (libnvinfer.so.10 etc.) for the ORT TensorRT EP (design D6 / task 5.6).
# The pip wheel drops the libs under site-packages/tensorrt_libs; put them on LD_LIBRARY_PATH.
RUN pip install --no-cache-dir "tensorrt-cu12==10.*"

ENV ORT_ROOT=/opt/onnxruntime
# CUDA math libs live in the pytorch base's pip nvidia wheels; expose ALL of them (ORT CUDA EP
# pulls cublas/cudnn/cufft/curand/cusparse/cusolver AND nvrtc/nvjitlink at load) + ORT + driver.
ENV NV=/opt/conda/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:\
/opt/conda/lib/python3.11/site-packages/tensorrt_libs:\
${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:\
${NV}/cuda_cupti/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:\
${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/nccl/lib:${NV}/nvtx/lib:\
/usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /workspace
CMD ["bash"]
