"""Single-case msprof executor for RMSNorm performance benchmarking."""

import argparse
import json
import torch
import torch_npu  # noqa: F401  Registers NPU device.
import ascend_kernel  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    case = json.loads(args.case)
    shape = tuple(case["shape"])
    dtype = getattr(torch, case["dtype"])
    eps = case["eps"]
    hidden_dim = shape[-1]

    x = torch.randn(shape, dtype=dtype, device="npu")
    w = torch.randn(hidden_dim, dtype=dtype, device="npu")

    # Warmup.
    for _ in range(args.warmup):
        _ = torch.ops.npu.rms_norm(x, w, eps)

    torch.npu.synchronize()

    # Timed iterations.
    for _ in range(args.iters - args.warmup):
        _ = torch.ops.npu.rms_norm(x, w, eps)

    torch.npu.synchronize()


if __name__ == "__main__":
    main()
