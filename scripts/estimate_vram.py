"""Estimate model-memory requirements without allocating a CUDA model.

This is a symbolic budget, not a benchmark.  It counts the tensors produced by
the frozen architectures for the registered input geometries and adds a
conservative activation envelope.  It deliberately never calls ``.cuda()``
and never trains or runs inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from partial_discharge_adaptive_fusion.modeling import make_expert


BYTES_PER_FP32 = 4
GIB = 2**30


def _temporal_trace(batch_size: int, signal_length: int) -> list[tuple[str, tuple[int, ...]]]:
    half = signal_length // 2
    quarter = signal_length // 4
    return [
        ("input", (batch_size, 1, signal_length)),
        ("conv1", (batch_size, 16, signal_length)),
        ("pool1", (batch_size, 16, half)),
        ("conv2", (batch_size, 32, half)),
        ("pool2", (batch_size, 32, quarter)),
        ("conv3", (batch_size, 64, quarter)),
        ("adaptive_avg_pool", (batch_size, 64, 1)),
    ]


def _cwt_trace(batch_size: int, scales: int = 32, time_bins: int = 120) -> list[tuple[str, tuple[int, ...]]]:
    return [
        ("input", (batch_size, 1, scales, time_bins)),
        ("conv1", (batch_size, 16, scales, time_bins)),
        ("pool1", (batch_size, 16, scales // 2, time_bins // 2)),
        ("conv2", (batch_size, 32, scales // 2, time_bins // 2)),
        ("pool2", (batch_size, 32, scales // 4, time_bins // 4)),
        ("conv3", (batch_size, 64, scales // 4, time_bins // 4)),
        ("adaptive_avg_pool", (batch_size, 64, 1, 1)),
    ]


def _bytes(shape: tuple[int, ...]) -> int:
    return math.prod(shape) * BYTES_PER_FP32


def estimate_case(name: str, kind: str, trace: list[tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    """Return a reproducible symbolic budget for one input/model case."""

    model = make_expert(kind)  # CPU construction only; no forward pass.
    parameters = sum(parameter.numel() for parameter in model.parameters())
    tensor_bytes = [_bytes(shape) for _, shape in trace]
    # Parameters + gradients + two FP32 Adam moments.
    optimizer_bytes = parameters * BYTES_PER_FP32 * 4
    activation_bytes = sum(tensor_bytes)
    return {
        "case": name,
        "expert": kind,
        "batch_size": trace[0][1][0],
        "parameter_count": parameters,
        "parameter_gradient_adam_bytes": optimizer_bytes,
        "input_shape": list(trace[0][1]),
        "largest_tensor_gib": max(tensor_bytes) / GIB,
        "all_traced_fp32_tensors_gib": activation_bytes / GIB,
        "three_x_activation_envelope_gib": 3 * activation_bytes / GIB,
        "model_payload_plus_three_x_activation_gib":
            (optimizer_bytes + 3 * activation_bytes) / GIB,
        "tensor_trace": [
            {"name": layer, "shape": list(shape), "fp32_gib": _bytes(shape) / GIB}
            for layer, shape in trace
        ],
    }


def estimate_all(batch_size: int = 128) -> dict[str, Any]:
    """Estimate all frozen MATLAB/VSB expert geometries."""

    return {
        "method": "symbolic_fp32_tensor_budget",
        "training": False,
        "cuda_allocation": False,
        "activation_envelope": "3x sum of traced input/output tensors; heuristic",
        "cuda_context_and_framework_overhead_included": False,
        "batch_size": batch_size,
        "parameter_counts": {
            "temporal": sum(p.numel() for p in make_expert("temporal").parameters()),
            "cwt": sum(p.numel() for p in make_expert("cwt").parameters()),
        },
        "cases": [
            estimate_case("matlab_temporal", "temporal", _temporal_trace(batch_size, 400)),
            estimate_case("vsb_temporal_native", "temporal", _temporal_trace(batch_size, 800_000)),
            estimate_case("matlab_cwt", "cwt", _cwt_trace(batch_size)),
            estimate_case("vsb_cwt_native_policy", "cwt", _cwt_trace(batch_size)),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    payload = estimate_all(args.batch_size)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
