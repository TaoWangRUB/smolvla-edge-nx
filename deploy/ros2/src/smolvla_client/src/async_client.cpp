// C++ async inference client — SmolVLA §3.3 Algorithm 1, a member-for-member port of
// src/smolvla_edge/async_infer.py::AsyncRunner (openspec: ros2-cpp-async-deployment, D4).
//
// One control tick per received /observation: the bridge owns the 50 Hz clock (design D3),
// so this node is observation-driven — one SimObservation == one tick == at most one action
// popped and published. Two unsynchronized wall timers would drift in phase; this can't.
//
// Differences from the Python runner, both deliberate:
//  - No virtual-time emulation: AsyncRunner delays a chunk's visibility by ceil(L/dt) ticks
//    to emulate latency inside a faster-than-realtime sim. Here the sim runs at wall-clock
//    50 Hz behind the bridge, so a chunk becomes visible when its gRPC reply actually lands.
//  - start_episode doesn't block: the Python runner blocks on the first chunk; here the
//    bridge holds the sim until the first action arrives, which is the same thing.
//
// At most one PredictChunk in flight (single worker thread), same as the Python pool(1).

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "policy.grpc.pb.h"
#include "smolvla_client/algorithm.hpp"
#include "smolvla_msgs/msg/policy_chunk.hpp"
#include "smolvla_msgs/msg/policy_request.hpp"
#include "smolvla_msgs/msg/sim_observation.hpp"
#include "smolvla_msgs/msg/tick_event.hpp"

using smolvla_msgs::msg::PolicyChunk;
using smolvla_msgs::msg::PolicyRequest;
using smolvla_msgs::msg::SimObservation;
using smolvla_msgs::msg::TickEvent;
using Clock = std::chrono::steady_clock;

namespace
{

smolvla_edge::Observation to_proto(const SimObservation & msg, const std::string & task)
{
  smolvla_edge::Observation obs;
  auto * img = obs.add_images();
  img->set_key("pixels.top");
  img->set_data(msg.image_top.data.data(), msg.image_top.data.size());
  img->add_shape(static_cast<int>(msg.image_top.height));
  img->add_shape(static_cast<int>(msg.image_top.width));
  img->add_shape(3);
  img->set_encoding("raw_uint8");
  auto * st = obs.add_tensors();
  st->set_key("agent_pos");
  for (float v : msg.agent_pos) {st->add_data(v);}
  st->add_shape(static_cast<int>(msg.agent_pos.size()));
  obs.set_task(task);
  return obs;
}

std::vector<double> joint_state(const SimObservation & msg)
{
  return {msg.agent_pos.begin(), msg.agent_pos.end()};
}

}  // namespace

class AsyncClient : public rclcpp::Node
{
public:
  AsyncClient()
  : Node("async_client")
  {
    server_ = declare_parameter<std::string>("server", "policy-server:50051");
    task_ = declare_parameter<std::string>("task", "");
    g_ = declare_parameter<double>("g", 0.7);
    epsilon_ = declare_parameter<double>("epsilon", 0.0);
    aggregate_ = declare_parameter<std::string>("aggregate", "new_wins");
    ramp_in_ = static_cast<int>(declare_parameter<int64_t>("ramp_in", 0));
    transport_ = declare_parameter<std::string>("transport", "grpc");

    if (transport_ == "grpc") {
      auto channel = grpc::CreateChannel(server_, grpc::InsecureChannelCredentials());
      stub_ = smolvla_edge::Policy::NewStub(channel);
      smolvla_edge::HealthRequest hreq;
      smolvla_edge::HealthReply hrep;
      grpc::ClientContext hctx;
      hctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
      auto status = stub_->Health(&hctx, hreq, &hrep);
      if (!status.ok()) {
        throw std::runtime_error("policy server unreachable at " + server_ + ": " +
                status.error_message());
      }
      RCLCPP_INFO(get_logger(), "policy server %s: device=%s precision=%s g=%.2f epsilon=%.3f",
        server_.c_str(), hrep.device().c_str(), hrep.precision().c_str(), g_, epsilon_);
    } else if (transport_ == "ros2") {
      // all-ROS2 policy hop: PolicyRequest/PolicyChunk over DDS (policy_node.py serves)
      rclcpp::QoS pqos(10);
      pqos.reliable();
      req_pub_ = create_publisher<PolicyRequest>("/policy/request", pqos);
      chunk_sub_ = create_subscription<PolicyChunk>(
        "/policy/chunk", pqos, [this](PolicyChunk::UniquePtr msg) {
          {
            std::lock_guard<std::mutex> lk(mtx_);
            ros2_reply_ = *msg;
          }
          cv_.notify_all();
        });
      // reliable QoS only guarantees delivery to MATCHED endpoints: block until the policy
      // node is discovered, or the first trigger's request would vanish (pending_ deadlock)
      const auto t0 = Clock::now();
      while (rclcpp::ok() &&
        (req_pub_->get_subscription_count() == 0 || chunk_sub_->get_publisher_count() == 0))
      {
        if (Clock::now() - t0 > std::chrono::seconds(60)) {
          throw std::runtime_error("ros2 policy node not discovered on /policy/* within 60s");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      RCLCPP_INFO(get_logger(), "ros2 policy node discovered: g=%.2f epsilon=%.3f", g_, epsilon_);
    } else {
      throw std::runtime_error("unknown transport: " + transport_);
    }

    rclcpp::QoS qos(1);
    qos.reliable();
    action_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/action", qos);
    event_pub_ = create_publisher<TickEvent>("/events", rclcpp::QoS(100).reliable());
    obs_sub_ = create_subscription<SimObservation>(
      "/observation", qos, [this](SimObservation::UniquePtr msg) {on_observation(*msg);});

    worker_ = std::thread([this] {worker_loop();});
  }

  ~AsyncClient() override
  {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {worker_.join();}
  }

private:
  struct Pending
  {
    int64_t trigger_tick;
    int pops;
    Clock::time_point t0;
  };
  struct Result
  {
    int64_t epoch;
    smolvla::Chunk chunk;
    double secs;
  };

  // -- one control tick (== AsyncRunner.act) ----------------------------------------------

  void on_observation(const SimObservation & msg)
  {
    const auto proc_t0 = Clock::now();
    // new episode == episode counter change only: the bridge re-publishes the tick-0
    // observation during the cold-start wait, and those must NOT reset our pending request
    if (msg.episode != episode_) {
      start_episode(msg);
    }
    TickEvent ev;
    ev.tick = msg.tick;
    ev.episode = msg.episode;
    ev.queue_depth = static_cast<int32_t>(queue_.size());
    maybe_merge(ev);

    if (queue_.empty()) {
      // empty queue: latest obs is processed regardless of similarity
      if (!pending_) {
        trigger(msg, /*forced=*/true, ev);
      }
      maybe_merge(ev);  // may have landed exactly this tick
    }

    if (!queue_.empty()) {
      std::vector<float> action = std::move(queue_.front());
      queue_.pop_front();
      last_action_ = action;
      if (pending_) {
        // an old-queue action executed after the pending obs was captured
        pending_->pops += 1;
      } else if (n_ > 0 && static_cast<double>(queue_.size()) / n_ < g_) {
        trigger(msg, /*forced=*/false, ev);  // threshold check after the pop (line 6)
      }
      std_msgs::msg::Float32MultiArray out;
      out.data = action;
      action_pub_->publish(out);
    } else {
      ev.idle = true;  // bridge holds pose this tick
    }

    ev.queue_after = static_cast<int32_t>(queue_.size());
    ev.proc_ms = std::chrono::duration<double, std::milli>(Clock::now() - proc_t0).count();
    event_pub_->publish(ev);
  }

  void start_episode(const SimObservation & msg)
  {
    episode_ = msg.episode;
    epoch_ += 1;  // stale in-flight results/requests from the previous episode are discarded
    queue_.clear();
    pending_.reset();
    last_action_.reset();
    last_sent_state_.reset();
    n_ = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      request_.reset();
      result_.reset();
    }
    RCLCPP_INFO(get_logger(), "episode %ld", static_cast<long>(episode_));
  }

  void trigger(const SimObservation & msg, bool forced, TickEvent & ev)
  {
    if (!forced && epsilon_ > 0 && last_sent_state_) {
      auto s = joint_state(msg);
      if (smolvla::joint_distance(s, *last_sent_state_) < epsilon_) {
        ev.filtered = true;
        return;
      }
    }
    // pops: threshold triggers fire right after a pop that supersedes the chunk's first
    // entry, hence 1; empty-queue (forced) triggers start at 0 — as in AsyncRunner._trigger
    pending_ = Pending{msg.tick, forced ? 0 : 1, Clock::now()};
    last_sent_state_ = joint_state(msg);
    {
      std::lock_guard<std::mutex> lk(mtx_);
      request_ = std::make_pair(epoch_, msg);  // transport-specific encoding in the worker
    }
    cv_.notify_one();
    ev.sent = true;
  }

  void maybe_merge(TickEvent & ev)
  {
    if (!pending_) {return;}
    std::optional<Result> res;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (result_ && result_->epoch == epoch_) {
        res = std::move(result_);
        result_.reset();
      } else if (result_) {
        result_.reset();  // stale episode
      }
    }
    if (!res) {return;}

    const int pops = pending_->pops;
    pending_.reset();
    smolvla::Chunk & chunk = res->chunk;
    if (n_ == 0) {n_ = static_cast<int>(chunk.rows);}  // chunk size from the first chunk
    if (static_cast<std::size_t>(pops) >= chunk.rows) {
      return;  // every timestep of the chunk was overtaken by executed actions
    }
    smolvla::Chunk fresh;
    fresh.rows = chunk.rows - pops;
    fresh.dim = chunk.dim;
    fresh.data.assign(chunk.row(pops), chunk.row(pops) + fresh.rows * fresh.dim);

    smolvla::Chunk old_q;
    old_q.rows = queue_.size();
    old_q.dim = fresh.dim;
    for (const auto & row : queue_) {
      old_q.data.insert(old_q.data.end(), row.begin(), row.end());
    }
    smolvla::Chunk merged = smolvla::aggregate_chunks(old_q, fresh, aggregate_);
    if (ramp_in_ > 0 && last_action_) {
      smolvla::ramp_in(merged, *last_action_, ramp_in_);
    }
    queue_.clear();
    for (std::size_t r = 0; r < merged.rows; ++r) {
      queue_.emplace_back(merged.row(r), merged.row(r) + merged.dim);
    }
    ev.merged = true;
    ev.rtt_s = res->secs;
  }

  // -- worker: single thread, at most one PredictChunk in flight ---------------------------

  void worker_loop()
  {
    while (true) {
      std::pair<int64_t, SimObservation> req;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] {return stop_ || request_.has_value();});
        if (stop_) {return;}
        req = std::move(*request_);
        request_.reset();
      }
      const auto t0 = Clock::now();
      Result res;
      res.epoch = req.first;
      if (transport_ == "grpc") {
        smolvla_edge::ActionChunk reply;
        grpc::ClientContext ctx;
        auto status = stub_->PredictChunk(&ctx, to_proto(req.second, task_), &reply);
        if (!status.ok()) {
          RCLCPP_ERROR(get_logger(), "PredictChunk failed: %s",
            status.error_message().c_str());
          continue;  // pending_ stays set; a failed RPC is fatal for the episode anyway
        }
        res.chunk.rows = reply.shape(0);
        res.chunk.dim = reply.shape(1);
        res.chunk.data.assign(reply.data().begin(), reply.data().end());
      } else {
        PolicyRequest preq;
        preq.request_id = ++ros2_req_id_;
        preq.task = task_;
        preq.image_top = req.second.image_top;
        preq.agent_pos = req.second.agent_pos;
        preq.client_send_ts =
          std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
          .count();
        req_pub_->publish(preq);
        std::unique_lock<std::mutex> lk(mtx_);
        const bool ok = cv_.wait_for(lk, std::chrono::seconds(60), [this] {
            return stop_ || (ros2_reply_ && ros2_reply_->request_id == ros2_req_id_);
          });
        if (stop_) {return;}
        if (!ok || !ros2_reply_) {
          RCLCPP_ERROR(get_logger(), "ros2 PredictChunk timed out (request %ld)",
            static_cast<long>(ros2_req_id_));
          continue;
        }
        res.chunk.rows = ros2_reply_->rows;
        res.chunk.dim = ros2_reply_->cols;
        res.chunk.data.assign(ros2_reply_->data.begin(), ros2_reply_->data.end());
        ros2_reply_.reset();
      }
      res.secs = std::chrono::duration<double>(Clock::now() - t0).count();
      {
        std::lock_guard<std::mutex> lk(mtx_);
        result_ = std::move(res);
      }
    }
  }

  // parameters
  std::string server_, task_, aggregate_, transport_;
  double g_{0.7}, epsilon_{0.0};
  int ramp_in_{0};

  // Algorithm-1 state (main thread only)
  std::deque<std::vector<float>> queue_;
  int n_{0};
  int64_t episode_{-1};
  int64_t epoch_{0};
  std::optional<Pending> pending_;
  std::optional<std::vector<float>> last_action_;
  std::optional<std::vector<double>> last_sent_state_;

  // worker plumbing
  std::unique_ptr<smolvla_edge::Policy::Stub> stub_;
  std::thread worker_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_{false};
  std::optional<std::pair<int64_t, SimObservation>> request_;
  std::optional<Result> result_;
  int64_t ros2_req_id_{0};
  std::optional<PolicyChunk> ros2_reply_;
  rclcpp::Publisher<PolicyRequest>::SharedPtr req_pub_;
  rclcpp::Subscription<PolicyChunk>::SharedPtr chunk_sub_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr action_pub_;
  rclcpp::Publisher<TickEvent>::SharedPtr event_pub_;
  rclcpp::Subscription<SimObservation>::SharedPtr obs_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AsyncClient>());
  rclcpp::shutdown();
  return 0;
}
