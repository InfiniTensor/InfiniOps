"""Compare warm Ascend operator latency using the selected InfiniOps package."""

import argparse
import json
import statistics
import time
from pathlib import Path

import infini.ops
import torch
import torch_npu  # noqa: F401


def measure(invoke, iterations, repeats):
    for _ in range(5):
        invoke()

    torch.npu.synchronize()
    samples = []

    for _ in range(repeats):
        start = time.perf_counter()

        for _ in range(iterations):
            invoke()

        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)

    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case-prefix", default="")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(41)
    torch.set_num_threads(2)
    stream = torch.npu.current_stream().npu_stream
    options = {"implementation_index": 0, "stream": stream}
    result = {
        "package": infini.ops.__file__,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "measurement": "warm host plus device latency; median of synchronized batches",
        "cases": {},
    }

    def record(name, invoke):
        if not name.startswith(args.case_prefix):
            return

        result["cases"][name] = measure(invoke, args.iterations, args.repeats)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, result["cases"][name]["median_us"], flush=True)

    input = torch.randn((64, 35, 128), dtype=torch.bfloat16, device="npu")[:, :32]
    weight = torch.ones(128, dtype=input.dtype, device=input.device)
    out = torch.empty((64, 32, 128), dtype=input.dtype, device=input.device)
    record(
        "rms_norm_bf16_64x32x128_packed",
        lambda: infini.ops.rms_norm(input, weight, 1e-6, out, **options),
    )

    qkv = torch.randn((64, 48, 128), dtype=torch.bfloat16, device="npu")
    query, key = qkv[:, :32], qkv[:, 32:40]
    positions = torch.arange(64, dtype=torch.int64, device="npu")
    cache = torch.cat(
        (torch.ones((128, 64), device="npu"), torch.zeros((128, 64), device="npu")),
        dim=1,
    ).bfloat16()
    record(
        "rotary_embedding_bf16_64x32_8x128_packed",
        lambda: infini.ops.rotary_embedding(
            positions, query, key, cache, 128, True, **options
        ),
    )

    for hidden, padded in ((3072, False), (6144, False), (3072, True), (6144, True)):
        input = torch.randn((64, hidden * 2), dtype=torch.bfloat16, device="npu")
        out = torch.empty_strided(
            (64, hidden),
            (hidden + 128 if padded else hidden, 1),
            dtype=input.dtype,
            device=input.device,
        )
        record(
            f"silu_and_mul_bf16_64x{hidden}{'_padded' if padded else ''}",
            lambda: infini.ops.silu_and_mul(input, out, **options),
        )

    for layout in ("NHD", "HND"):
        tokens, heads, dim, block_size = 8192, 8, 128, 128
        packed = torch.randn(
            (tokens, heads * 3, dim), dtype=torch.bfloat16, device="npu"
        )
        key, value = packed[:, :heads], packed[:, heads : heads * 2]
        shape = (
            (tokens // block_size, block_size, heads, dim)
            if layout == "NHD"
            else (tokens // block_size, heads, block_size, dim)
        )
        key_cache = torch.zeros(shape, dtype=key.dtype, device=key.device)
        value_cache = torch.zeros_like(key_cache)

        if layout == "HND":
            key_cache, value_cache = (
                key_cache.transpose(1, 2),
                value_cache.transpose(1, 2),
            )

        slots = torch.arange(tokens, dtype=torch.int64, device="npu")
        scale = torch.ones((), dtype=torch.float32, device="npu")
        record(
            f"reshape_and_cache_bf16_8192x8x128_{layout}",
            lambda: infini.ops.reshape_and_cache_flash(
                key,
                value,
                slots,
                scale,
                scale,
                "auto",
                key_cache,
                value_cache,
                **options,
            ),
        )


if __name__ == "__main__":
    main()
