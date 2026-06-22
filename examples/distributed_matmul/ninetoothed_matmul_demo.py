#!/usr/bin/env python3
import argparse
import time

import torch

with __import__('contextlib').suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401
with __import__('contextlib').suppress(ImportError, ModuleNotFoundError):
    import torch_musa  # noqa: F401
with __import__('contextlib').suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 32


def arrangement(lhs, rhs, output, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    lhs_tiled = lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K)).tile((1, -1)).expand((-1, output_tiled.shape[1]))
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled = rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N)).tile((-1, 1)).expand((output_tiled.shape[0], -1))
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)
    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


def available_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'mlu') and torch.mlu.is_available():
        return 'mlu'
    if hasattr(torch, 'musa') and torch.musa.is_available():
        return 'musa'
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    raise RuntimeError('no supported torch accelerator is available')


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mlu':
        torch.mlu.synchronize()
    elif device == 'musa':
        torch.musa.synchronize()
    elif device == 'npu':
        torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--atol', type=float, default=2e-2)
    parser.add_argument('--rtol', type=float, default=2e-2)
    args = parser.parse_args()

    device = available_device()
    torch.manual_seed(20240622)
    lhs = torch.randn((args.m, args.k), dtype=torch.float16, device=device)
    rhs = torch.randn((args.k, args.n), dtype=torch.float16, device=device)
    output = torch.empty((args.m, args.n), dtype=torch.float16, device=device)
    kernel = ninetoothed.make(arrangement, application, (Tensor(2), Tensor(2), Tensor(2)))

    for _ in range(args.warmup):
        kernel(lhs, rhs, output)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        kernel(lhs, rhs, output)
    sync(device)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / args.iters

    expected = torch.matmul(lhs, rhs)
    max_error = (output - expected).abs().max().item()
    ok = torch.allclose(output, expected, atol=args.atol, rtol=args.rtol)
    tflops = (2.0 * args.m * args.n * args.k) / (avg_ms / 1000.0) / 1.0e12

    print(f'torch={torch.__version__} device={device}')
    print(f'shape=[{args.m}, {args.n}] k={args.k} block=[16,16,32] dtype=float16 max_error={max_error:.6g} avg_ms={avg_ms:.6g} tflops={tflops:.6g} tolerance=atol:{args.atol},rtol:{args.rtol} status={"ok" if ok else "failed"}')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
