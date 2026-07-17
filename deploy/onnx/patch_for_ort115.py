"""Make the exported SmolVLA graph loadable by the Jetson's onnxruntime-gpu 1.15.1 (JetPack 5).

The monolithic export targets a newer ORT (>=1.18 on x86). Two things trip ORT 1.15.1, the
newest CUDA-11.4 / cp38 wheel available for JetPack 5:

  1. IR version 10 (ONNX >=1.16) — ORT 1.15 caps at IR 9. The graph uses only opset-17 ops,
     so downgrading the container version to 9 is safe.
  2. ArgMin/ArgMax with an INT64 input — ORT 1.15's CPU/CUDA kernels register ArgMin/ArgMax
     for int32/float/... but NOT int64, so the node raises NOT_IMPLEMENTED. We insert a
     Cast(int64 -> int32) in front of it. ArgMin returns the index of the min, which is
     unchanged by a value-preserving int64->int32 cast (the compared values are small
     token-position integers, well within int32).

Both are load-time compatibility shims, not numerical changes; the FP32 parity gate
(deploy/onnx/parity.py) still applies to the compute path.

    python3 deploy/onnx/patch_for_ort115.py \
        --in  models/onnx/smolvla_transfer_cube.onnx \
        --out models/onnx/smolvla_transfer_cube_ort115.onnx
"""

from __future__ import annotations

import argparse

import onnx
from onnx import TensorProto, helper


def _int64_typed_inputs(graph) -> set:
    """Names whose declared elem_type is INT64 (value_info + inputs + initializers)."""
    names = set()
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.INT64:
            names.add(vi.name)
    for init in graph.initializer:
        if init.data_type == TensorProto.INT64:
            names.add(init.name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    model = onnx.load(args.src)
    if model.ir_version > 9:
        print(f"[patch] ir_version {model.ir_version} -> 9")
        model.ir_version = 9

    g = model.graph
    int64_names = _int64_typed_inputs(g)

    new_nodes, patched = [], 0
    for n in g.node:
        if n.op_type in ("ArgMin", "ArgMax") and n.input and n.input[0] in int64_names:
            src = n.input[0]
            cast_out = f"{src}__i32_for_{n.name}"
            new_nodes.append(
                helper.make_node("Cast", [src], [cast_out], to=TensorProto.INT32,
                                 name=f"{n.name}__castpre")
            )
            n.input[0] = cast_out
            patched += 1
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    print(f"[patch] inserted {patched} int64->int32 Cast(s) before ArgMin/ArgMax")

    onnx.save(model, args.dst)
    print(f"[patch] wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
