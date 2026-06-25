# Xavier NX (JetPack) setup

The Jetson is its own world: aarch64 wheels, NVIDIA-provided torch, TensorRT baked into
JetPack, and power modes that materially change benchmark numbers. Do **not** `pip install
torch` from PyPI here.

## 1. Base

- Flash a recent JetPack (record the exact version — it pins the CUDA/cuDNN/TensorRT triple).
- Confirm: `nvcc --version`, `dpkg -l | grep -i tensorrt`.
- Max out clocks for benchmarking and record the mode:
  ```bash
  sudo nvpmodel -m 0      # max-N (record `nvpmodel -q`)
  sudo jetson_clocks
  ```

## 2. PyTorch

Install the NVIDIA aarch64 torch wheel matching your JetPack (from the NVIDIA Jetson PyTorch
index / forum), then torchvision built against it. Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. LeRobot + this repo

```bash
pip install -r requirements.txt   # torch is already installed; pip should not replace it
pip install -e .                  # exposes the smolvla_edge package
```
If `lerobot[smolvla]` tries to pull a PyPI torch, install LeRobot with `--no-deps` and add its
remaining deps manually so the NVIDIA torch stays put.

## 4. Memory headroom on 8 GB

- Prefer FP16. Mount swap/zram if you hit OOM during model load.
- Close the desktop GUI (`sudo systemctl isolate multi-user.target`) to reclaim VRAM.
- Watch live: `tegrastats` (or `jtop` from `jetson-stats`).

## 5. Sanity check

```bash
python -m smolvla_edge.infer --policy-path <checkpoint> --episodes 1 --max-frames 10
```

Then proceed to `deploy/ondevice/` (on-device) and the benchmark commands in
`benchmarks/README.md`.
