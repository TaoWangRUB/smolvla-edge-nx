// C++ ONNX Runtime inference server (Stage 2b, cpp-inference-server / design D6).
//
// Implements the SAME Policy gRPC service (PredictChunk / Reset / Health) the Python server.py
// serves, backed by the monolithic SmolVLA ONNX graph (deploy/onnx/export_smolvla.py) under
// ONNX Runtime's CUDA (or TensorRT) execution provider. Wire-compatible: the Python reference
// client and the ROS2 async_client run against it unmodified (task 5.4).
//
// The graph bakes in task tokens + normalization + the 10 unrolled Euler steps, so this server's
// only preprocessing is the serving-side resize_with_pad (spec: "raw resized images"): the raw
// uint8 HWC frame the client sends -> 512x512 [0,1] CHW, matching SmolVLA's prepare_images. The
// denoising noise is sampled here (the graph takes it as an explicit input).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "policy.grpc.pb.h"

using smolvla_edge::ActionChunk;
using smolvla_edge::HealthReply;
using smolvla_edge::HealthRequest;
using smolvla_edge::Observation;
using smolvla_edge::ResetReply;
using smolvla_edge::ResetRequest;

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// resize_with_pad matching lerobot SmolVLA prepare_images: keep aspect ratio to fit within
// (target x target), bilinear, then zero-pad on the TOP (and left) — here 480x640 -> 384x512
// -> pad 128 rows on top -> 512x512. Output is CHW float in [0,1], RGB order preserved.
std::vector<float> preprocess_image(const uint8_t* rgb, int h, int w, int target) {
  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(rgb));  // HWC, RGB
  double ratio = std::max(static_cast<double>(w) / target, static_cast<double>(h) / target);
  int rh = static_cast<int>(h / ratio), rw = static_cast<int>(w / ratio);
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
  cv::Mat padded(target, target, CV_8UC3, cv::Scalar(0, 0, 0));
  int top = target - rh, left = target - rw;  // pad on top/left
  resized.copyTo(padded(cv::Rect(left, top, rw, rh)));

  std::vector<float> chw(3 * target * target);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < target; ++y)
      for (int x = 0; x < target; ++x)
        chw[c * target * target + y * target + x] =
            padded.at<cv::Vec3b>(y, x)[c] / 255.0f;   // Vec3b is RGB (we built it RGB)
  return chw;
}

}  // namespace

class PolicyServiceImpl final : public smolvla_edge::Policy::Service {
 public:
  PolicyServiceImpl(const std::string& model_path, const std::string& provider)
      : env_(ORT_LOGGING_LEVEL_WARNING, "smolvla_cpp_server"),
        model_path_(model_path), provider_(provider), rng_(1234) {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "cuda" || provider == "tensorrt") {
      if (provider == "tensorrt") {
        OrtTensorRTProviderOptions trt{};
        trt.device_id = 0;
        trt.trt_fp16_enable = 0;
        trt.trt_engine_cache_enable = 1;
        trt.trt_engine_cache_path = "/workspace/models/onnx/trt_cache";
        opts.AppendExecutionProvider_TensorRT(trt);
      }
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cuda);   // TRT falls back to CUDA for unsupported nodes
    }
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_->GetInputCount(); ++i)
      input_names_.push_back(session_->GetInputNameAllocated(i, alloc).get());
    for (size_t i = 0; i < session_->GetOutputCount(); ++i)
      output_names_.push_back(session_->GetOutputNameAllocated(i, alloc).get());
    // chunk shape from the action_chunk output: [1, chunk, action_dim]
    auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    chunk_ = static_cast<int>(out_shape[1]);
    action_dim_ = static_cast<int>(out_shape[2]);
    // noise shape from the "noise" input: [1, chunk, max_action_dim]
    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (input_names_[i] == "noise") {
        auto s = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        noise_dim_ = static_cast<int>(s[2]);
      }
    }
    fprintf(stderr, "[cpp_server] loaded %s provider=%s chunk=%d action_dim=%d noise_dim=%d\n",
            model_path.c_str(), provider.c_str(), chunk_, action_dim_, noise_dim_);
  }

  grpc::Status PredictChunk(grpc::ServerContext*, const Observation* req,
                            ActionChunk* reply) override {
    double recv = now_s();
    if (req->images_size() == 0)
      return {grpc::StatusCode::INVALID_ARGUMENT, "no image in observation"};
    const auto& img = req->images(0);
    int h = img.shape(0), w = img.shape(1);
    auto image = preprocess_image(reinterpret_cast<const uint8_t*>(img.data().data()), h, w, kImg);

    std::vector<float> state(action_dim_, 0.f);
    if (req->tensors_size() > 0) {
      const auto& t = req->tensors(0);
      for (int i = 0; i < action_dim_ && i < t.data_size(); ++i) state[i] = t.data(i);
    }

    std::vector<float> noise(static_cast<size_t>(chunk_) * noise_dim_);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : noise) v = nd(rng_);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> img_shape{1, 3, kImg, kImg};
    std::vector<int64_t> st_shape{1, action_dim_};
    std::vector<int64_t> nz_shape{1, chunk_, noise_dim_};
    std::vector<Ort::Value> inputs;
    // input order must match input_names_ (image, state, noise)
    for (const auto& name : input_names_) {
      if (name == "image")
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, image.data(), image.size(),
                                                         img_shape.data(), img_shape.size()));
      else if (name == "state")
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, state.data(), state.size(),
                                                         st_shape.data(), st_shape.size()));
      else
        inputs.push_back(Ort::Value::CreateTensor<float>(mem, noise.data(), noise.size(),
                                                         nz_shape.data(), nz_shape.size()));
    }

    std::vector<const char*> in_names, out_names;
    for (auto& s : input_names_) in_names.push_back(s.c_str());
    for (auto& s : output_names_) out_names.push_back(s.c_str());
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                             inputs.size(), out_names.data(), out_names.size());
    const float* data = out[0].GetTensorData<float>();
    size_t n = static_cast<size_t>(chunk_) * action_dim_;
    reply->mutable_data()->Add(data, data + n);
    reply->add_shape(chunk_);
    reply->add_shape(action_dim_);
    reply->set_server_recv_ts(recv);
    reply->set_server_send_ts(now_s());
    return grpc::Status::OK;
  }

  grpc::Status Reset(grpc::ServerContext*, const ResetRequest*, ResetReply* reply) override {
    reply->set_ok(true);  // the graph is stateless
    return grpc::Status::OK;
  }

  grpc::Status Health(grpc::ServerContext*, const HealthRequest*, HealthReply* reply) override {
    reply->set_ok(true);
    reply->set_device(provider_ == "cpu" ? "cpu" : "cuda");
    reply->set_precision("fp32");
    reply->set_policy_path(model_path_ + " (" + provider_ + ")");
    return grpc::Status::OK;
  }

 private:
  static constexpr int kImg = 512;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string model_path_, provider_;
  std::vector<std::string> input_names_, output_names_;
  int chunk_{50}, action_dim_{14}, noise_dim_{32};
  std::mt19937 rng_;
};

int main(int argc, char** argv) {
  std::string model = "/workspace/models/onnx/smolvla_transfer_cube.onnx";
  std::string provider = "cuda";
  int port = 50051;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--provider" && i + 1 < argc) provider = argv[++i];
    else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
  }
  PolicyServiceImpl service(model, provider);
  std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc::ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  fprintf(stderr, "[cpp_server] listening on %s\n", addr.c_str());
  server->Wait();
  return 0;
}
