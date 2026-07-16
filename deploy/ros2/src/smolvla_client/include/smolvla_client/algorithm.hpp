// Chunk aggregation + similarity filter — line-for-line port of
// src/smolvla_edge/async_infer.py (aggregate_chunks, _joint_state distance, ramp_in).
// Header-only and ROS-free so the gtest can validate it against fixtures exported from the
// Python implementation (task 3.5).
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace smolvla
{

// Row-major [rows x dim] action matrix.
struct Chunk
{
  std::size_t rows{0};
  std::size_t dim{0};
  std::vector<float> data;  // rows * dim

  float * row(std::size_t r) {return data.data() + r * dim;}
  const float * row(std::size_t r) const {return data.data() + r * dim;}
};

// f(A_t, A~_t+1): combine the remaining old queue with the (aligned) new chunk. Both start
// at the current tick; the result has new_chunk.rows rows (async_infer.aggregate_chunks).
inline Chunk aggregate_chunks(const Chunk & old_q, const Chunk & new_c, const std::string & how)
{
  if (how == "new_wins" || old_q.rows == 0) {
    return new_c;
  }
  if (how != "blend") {
    throw std::invalid_argument("unknown aggregator '" + how + "' (want new_wins|blend)");
  }
  const std::size_t m = std::min(old_q.rows, new_c.rows);
  Chunk out = new_c;
  for (std::size_t i = 0; i < m; ++i) {
    // np.linspace(0.5, 1.0, m): w_i = 0.5 + 0.5 * i / (m - 1); single point -> 0.5
    const float w = (m == 1) ? 0.5f : 0.5f + 0.5f * static_cast<float>(i) / (m - 1);
    for (std::size_t j = 0; j < new_c.dim; ++j) {
      out.row(i)[j] = (1.0f - w) * old_q.row(i)[j] + w * new_c.row(i)[j];
    }
  }
  return out;
}

// ramp_in: blend the first k post-merge actions from the last executed action (linear),
// w_i = (i+1)/(k+1) for i in [0, k) — matches np.linspace(1/(k+1), k/(k+1), k).
inline void ramp_in(Chunk & merged, const std::vector<float> & last_action, int k_max)
{
  if (k_max <= 0 || last_action.empty() || merged.rows == 0) {
    return;
  }
  const std::size_t k = std::min<std::size_t>(k_max, merged.rows);
  for (std::size_t i = 0; i < k; ++i) {
    const float w = static_cast<float>(i + 1) / static_cast<float>(k + 1);
    for (std::size_t j = 0; j < merged.dim; ++j) {
      merged.row(i)[j] = (1.0f - w) * last_action[j] + w * merged.row(i)[j];
    }
  }
}

// Joint-space L2 distance for the epsilon similarity filter (float64, as in _joint_state).
inline double joint_distance(const std::vector<double> & a, const std::vector<double> & b)
{
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

}  // namespace smolvla
