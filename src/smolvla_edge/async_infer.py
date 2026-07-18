"""Asynchronous inference runtime — SmolVLA paper §3.3, Algorithm 1.

A RobotClient-side control loop that consumes actions from a queue while chunk
prediction runs on a background worker (the in-process "PolicyServer"):

  - pop one action per control tick; when the remaining queue fraction drops below
    the threshold ``g``, capture the current observation and trigger a NON-BLOCKING
    chunk prediction (at most one in flight);
  - drop near-duplicate observations (joint-space distance < ``epsilon``), except
    when the queue is empty, where the latest observation is always processed;
  - when a new chunk arrives, aggregate it with the live queue on the overlapping
    timesteps via a pluggable ``f(A_t, A~_t+1)``.

Sync inference is the ``g = 0`` limit of the same loop: the queue drains fully,
then a (forced) prediction is triggered and the client idles until it lands.

Timing model — the crux for simulation. gym-aloha steps as fast as the CPU allows
and would happily freeze the world while the policy thinks, which hides exactly
the lag async inference removes. We therefore run in *virtual time*: one env.step
== one control tick of length ``dt`` (20 ms at ALOHA's 50 Hz), and a chunk whose
prediction took ``L`` wall-clock seconds becomes visible ``ceil(L / dt)`` ticks
after it was triggered. Ticks where the queue is empty and no chunk has landed
are *idle ticks* — the sim analogue of the robot standing still — and the caller
executes a hold action for them. ``idle="freeze"`` disables the emulation
(chunks land the moment they finish, no idle injected), reproducing the classic
frozen-env eval for comparability with the Phase-1 numbers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

AGGREGATORS = ("new_wins", "blend")


def aggregate_chunks(old: np.ndarray, new: np.ndarray, how: str) -> np.ndarray:
    """f(A_t, A~_t+1): combine the remaining old queue with the (aligned) new chunk.

    Both arrays start at the *current* tick. The new chunk always extends at least
    as far as the old queue (it was predicted from a later observation), so the
    result has ``len(new)`` rows.

      - ``new_wins``: the incoming chunk replaces the overlap (paper default spirit).
      - ``blend``: convex combination over the overlap, weight ramping linearly
        from 0.5 (trust both now) to 1.0 (trust only the fresher chunk later) —
        the old chunk's late actions are its least reliable ones.
    """
    if how == "new_wins" or len(old) == 0:
        return new
    if how != "blend":
        raise ValueError(f"unknown aggregator {how!r} (want one of {AGGREGATORS})")
    m = min(len(old), len(new))
    w = np.linspace(0.5, 1.0, m, dtype=new.dtype)[:, None]
    out = new.copy()
    out[:m] = (1.0 - w) * old[:m] + w * new[:m]
    return out


class AsyncRunner:
    """Algorithm 1 control loop. Call :meth:`start_episode` once, then
    :meth:`act` once per control tick; ``None`` means idle (caller holds pose)."""

    def __init__(
        self,
        predict_chunk,  # obs -> np.ndarray (chunk_len, action_dim); runs on the worker
        g: float = 0.7,
        epsilon: float = 0.0,  # 0 disables the similarity filter
        dt: float = 0.02,  # control period; 50 Hz for ALOHA
        aggregate: str = "new_wins",
        idle: str = "hold",  # "hold" = virtual-time idle emulation, "freeze" = block
        state_key: str = "agent_pos",
        ramp_in: int = 0,  # blend this many post-merge actions from the last
        # executed action (linear), smoothing the position-target step a splice
        # introduces when consecutive chunks disagree
    ):
        if not 0.0 <= g <= 1.0:
            raise ValueError(f"g must be in [0, 1], got {g}")
        if idle not in ("hold", "freeze"):
            raise ValueError(f"idle must be 'hold' or 'freeze', got {idle!r}")
        self._predict = predict_chunk
        self.g, self.epsilon, self.dt = g, epsilon, dt
        self.aggregate, self.idle_mode, self.state_key = aggregate, idle, state_key
        self.ramp_in = ramp_in
        # single worker: serializes all predictions, so the policy is never
        # called concurrently and at most one chunk is in flight
        self._pool = ThreadPoolExecutor(max_workers=1)

    # -- episode lifecycle ---------------------------------------------------

    def start_episode(self, obs) -> None:
        """Blocking cold-start predict (Algorithm 1 line 2: receive A_0 first)."""
        self.tick = 0
        self.queue: deque = deque()
        self._pending = None
        self._last_sent_state = None
        self._last_action = None
        self.trace: list[dict] = []
        self.stats = {"obs_sent": 0, "obs_filtered": 0, "merges": 0, "idle_ticks": 0,
                      "latencies_s": []}
        chunk, secs = self._pool.submit(self._timed_predict, obs).result()
        self.n = len(chunk)  # chunk size, denominator of the g threshold
        self.queue.extend(chunk)
        self.stats["latencies_s"].append(secs)
        self._remember_state(obs)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    # -- one control tick ----------------------------------------------------

    def act(self, obs):
        """Returns (action | None, event-dict). None == idle tick: hold pose."""
        ev = {"tick": self.tick, "queue": len(self.queue), "sent": False,
              "filtered": False, "merged": False, "idle": False}
        self._maybe_merge(ev)

        if not self.queue:
            # empty queue: latest obs is processed regardless of similarity
            if self._pending is None:
                self._trigger(obs, forced=True, ev=ev)
            if self.idle_mode == "freeze":
                self._block_merge(ev)
            else:
                self._maybe_merge(ev)  # may have landed exactly this tick

        if self.queue:
            action = self.queue.popleft()
            self._last_action = action
            if self._pending is not None:
                # an old-queue action executed after the pending obs was captured:
                # the incoming chunk overlaps it by one more timestep
                self._pending["pops"] += 1
            # threshold check after the pop (Algorithm 1 line 6)
            elif len(self.queue) / self.n < self.g:
                self._trigger(obs, forced=False, ev=ev)
        else:
            action = None
            ev["idle"] = True
            self.stats["idle_ticks"] += 1

        ev["queue_after"] = len(self.queue)
        self.trace.append(ev)
        self.tick += 1
        return action, ev

    # -- internals -------------------------------------------------------------

    def _timed_predict(self, obs):
        t0 = time.perf_counter()
        chunk = np.asarray(self._predict(obs))
        return chunk, time.perf_counter() - t0

    def _joint_state(self, obs):
        s = obs.get(self.state_key, obs.get("qpos")) if isinstance(obs, dict) else None
        return None if s is None else np.asarray(s, dtype=np.float64).ravel()

    def _remember_state(self, obs) -> None:
        self._last_sent_state = self._joint_state(obs)

    def _trigger(self, obs, forced: bool, ev: dict) -> None:
        if not forced and self.epsilon > 0 and self._last_sent_state is not None:
            s = self._joint_state(obs)
            if s is not None and np.linalg.norm(s - self._last_sent_state) < self.epsilon:
                ev["filtered"] = True
                self.stats["obs_filtered"] += 1
                return
        # "pops" counts actions executed since this observation was captured — the
        # overlap to skip in the incoming chunk. Idle (hold) ticks do NOT count:
        # the robot doesn't advance along the trajectory while holding pose.
        # Threshold triggers fire right after a pop whose action supersedes the
        # chunk's first entry, hence 1; empty-queue triggers start at 0.
        self._pending = {"tick": self.tick, "t_submit": time.perf_counter(),
                         "pops": 0 if forced else 1,
                         "future": self._pool.submit(self._timed_predict, obs)}
        self._remember_state(obs)
        ev["sent"] = True
        self.stats["obs_sent"] += 1

    def _available_tick(self, secs: float, trigger_tick: int) -> int:
        if self.idle_mode == "freeze":
            return trigger_tick  # land as soon as it's done
        return trigger_tick + max(1, math.ceil(secs / self.dt))

    def _maybe_merge(self, ev: dict) -> None:
        p = self._pending
        if p is None:
            return
        if not p["future"].done():
            if self.idle_mode == "freeze":
                return  # Algorithm 1 line 14-16: keep the current queue
            # Virtual-time honesty: sim ticks may outrun the wall clock (env.step
            # is often much faster than dt). If tick-time has caught up with the
            # real elapsed prediction time, pause the sim clock (block) so the
            # chunk's arrival tick reflects its true latency, not env speed.
            elapsed = time.perf_counter() - p["t_submit"]
            if self.tick < p["tick"] + max(1, math.ceil(elapsed / self.dt)):
                return  # definitely not arrived yet in tick time
        chunk, secs = p["future"].result()  # done, or sim clock paused until done
        if self.tick < self._available_tick(secs, p["tick"]):
            return  # finished, but hasn't "arrived" yet in virtual time
        self._merge(p, chunk, secs, ev)

    def _block_merge(self, ev: dict) -> None:
        p = self._pending
        if p is None:
            return
        chunk, secs = p["future"].result()  # blocks
        self._merge(p, chunk, secs, ev)

    def _merge(self, p: dict, chunk: np.ndarray, secs: float, ev: dict) -> None:
        self._pending = None
        self.stats["latencies_s"].append(secs)
        # skip the timesteps already covered by actions executed since the
        # observation was captured (idle ticks don't advance the trajectory)
        fresh = chunk[p["pops"]:]
        if len(fresh) == 0:
            return  # every timestep of the chunk was overtaken by executed actions
        old = np.asarray(self.queue) if self.queue else np.empty((0,) + fresh.shape[1:])
        merged = aggregate_chunks(old, fresh, self.aggregate)
        if self.ramp_in > 0 and self._last_action is not None:
            k = min(self.ramp_in, len(merged))
            w = np.linspace(1.0 / (k + 1), k / (k + 1.0), k)[:, None]
            merged = merged.copy()
            merged[:k] = (1.0 - w) * np.asarray(self._last_action) + w * merged[:k]
        self.queue = deque(merged)
        ev["merged"] = True
        ev["latency_s"] = secs
        self.stats["merges"] += 1

    # -- reporting -------------------------------------------------------------

    def episode_stats(self) -> dict:
        import statistics

        lats = self.stats["latencies_s"]
        return {
            "median_latency_s": statistics.median(lats) if lats else float("nan"),
            "ticks": self.tick,
            "chunk_size": self.n,
            "g": self.g,
            "epsilon": self.epsilon,
            "obs_sent": self.stats["obs_sent"],
            "obs_filtered": self.stats["obs_filtered"],
            "merges": self.stats["merges"],
            "idle_ticks": self.stats["idle_ticks"],
            "mean_latency_s": sum(lats) / len(lats) if lats else float("nan"),
        }


def make_chunk_predictor(policy, policy_path: str, device: str, task: str | None,
                         precision: str = "fp32"):
    """Chunk-level analogue of ``eval.make_sim_stepper``: obs -> full action chunk
    (chunk_len, action_dim) via ``policy.predict_action_chunk``, with the same
    two checkpoint formats (saved processor pipeline vs. manual mapping).

    ``precision="fp16"`` wraps the forward in CUDA autocast — the first lever for
    cutting server latency l_S when it approaches the chunk duration n*dt.
    ``precision="fp16-graph"`` additionally halves the policy and lazily captures the
    full ``sample_actions`` forward as one CUDA graph (the launch-bound fix — 608 ->
    233 ms/chunk on the Xavier NX, bitwise-identical actions; see
    ``smolvla_edge.cuda_graph``). Falls back to plain fp16 if capture fails."""
    import torch

    from .common import resolve_policy_path
    from .eval import _aloha_obs_to_batch, _batchify_obs, _load_normalizers

    policy_path = resolve_policy_path(policy_path)

    on_cuda = str(device).startswith("cuda")
    use_amp = precision in ("fp16", "fp16-graph") and on_cuda
    if precision in ("fp16", "fp16-graph") and not use_amp:
        print(f"[async] {precision} requested but device is not CUDA — running fp32")
    if precision == "fp16-graph" and on_cuda:
        policy.eval()
        policy.half()  # fp16 tensor cores — the config the capture was validated against
        from .cuda_graph import enable_lazy_graph

        enable_lazy_graph(policy)

    def _forward(batch):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                             enabled=use_amp):
            return policy.predict_action_chunk(batch)

    try:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors

        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        def predict_pipeline(obs):
            o = preprocess_observation(_batchify_obs(obs))
            if task:
                o["task"] = [task]
            o = pre(o)
            chunk = _forward(o)
            return post(chunk.float()).squeeze(0).float().cpu().numpy()

        print("[async] chunk predictor: saved processor pipeline")
        return predict_pipeline
    except Exception as e:
        print(f"[async] no processor pipeline ({type(e).__name__}); manual obs mapping")

    normalize_obs, unnormalize_action = _load_normalizers(policy, policy_path, device)
    from .common import make_language_tokenizer

    tokenize = make_language_tokenizer(policy, device)

    def predict_manual(obs):
        batch = normalize_obs(_aloha_obs_to_batch(obs, device, task))
        if tokenize is not None:
            batch.update(tokenize(task or "do the task"))
        chunk = unnormalize_action(_forward(batch).float())
        return chunk.squeeze(0).float().cpu().numpy()

    return predict_manual


def _selftest() -> None:
    """No-GPU sanity check of the Algorithm-1 loop with a fake 40 ms predictor.

    Expects, at 50 Hz / n=20 / g=0.5: a sawtooth queue trace with zero idle ticks
    (threshold triggers with 10 actions ~ 200 ms of buffer >> 40 ms latency), and
    idle ticks == ceil(latency/dt) per cycle in the g=0 sequential limit.
    """
    n, dim, lat = 20, 4, 0.04

    def fake_predict(obs):
        time.sleep(lat)
        return np.tile(np.asarray(obs["agent_pos"], dtype=np.float32), (n, 1))

    for g, want_idle_free in ((0.5, True), (0.0, False)):
        r = AsyncRunner(fake_predict, g=g, epsilon=0.0, dt=0.02)
        r.start_episode({"agent_pos": np.zeros(dim)})
        for t in range(200):
            a, ev = r.act({"agent_pos": np.full(dim, float(t))})
            time.sleep(0.001)  # pretend env.step takes ~1 ms
        s = r.episode_stats()
        r.close()
        print(f"g={g}: {s}")
        assert (s["idle_ticks"] == 0) == want_idle_free, s
        assert s["merges"] >= 1 and s["obs_sent"] >= 1, s
    print("[async] selftest OK")


if __name__ == "__main__":
    _selftest()
