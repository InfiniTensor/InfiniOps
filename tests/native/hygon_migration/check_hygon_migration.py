"""Compare migrated kernels against preserved kernels and integer references."""

import argparse
import ctypes
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(args.library)
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    size = ctypes.c_size_t
    lib.check_route.argtypes = [ptr, ptr, ptr, ptr, size, ctypes.c_float, integer, ptr]
    lib.check_quant.argtypes = [ptr, ptr, ptr, size, integer, integer, integer, ptr]
    lib.check_gemv.argtypes = [ptr] * 6 + [integer] * 4 + [ctypes.c_ssize_t, ptr]
    for name in ['check_route', 'check_quant', 'check_gemv']:
        getattr(lib, name).restype = integer
    torch.manual_seed(20261124)
    counts = {'route': 0, 'quant': 0, 'gemv': 0, 'graph_replays': 0}
    def stream():
        return ptr(torch.cuda.current_stream().cuda_stream)
    def addr(x):
        return ptr(x.data_ptr()) if x is not None else None
    def exact(a, b):
        if a.is_floating_point():
            assert torch.equal(torch.isnan(a), torch.isnan(b))
            mask = ~torch.isnan(a)
            if a.dtype == torch.float32:
                assert torch.equal(a.view(torch.int32)[mask], b.view(torch.int32)[mask])
            else:
                assert torch.equal(a.view(torch.int16)[mask], b.view(torch.int16)[mask])
        else:
            assert torch.equal(a, b)

    for rows in [1, 3, 4, 5, 31, 64, 257]:
        for pattern in ['random', 'ties', 'negative', 'nan', 'infinity', 'signed_zero']:
            x = torch.randn((rows, 256), device='cuda')
            bias = torch.randn(256, device='cuda') * 0.1
            if pattern == 'ties':
                x.zero_(); bias.zero_()
            elif pattern == 'negative':
                x.fill_(-80); bias.zero_()
            elif pattern == 'nan':
                x[:, 3] = float('nan')
            elif pattern == 'infinity':
                x[:, 5] = float('inf'); bias[7] = -float('inf')
            elif pattern == 'signed_zero':
                x.fill_(-float('inf')); bias.zero_(); bias[::2] = -0.0
            for use_bias in [False, True]:
                b = bias if use_bias else None
                w = [torch.empty((rows, 6), device='cuda') for _ in range(2)]
                ids = [torch.empty((rows, 6), device='cuda', dtype=torch.int64) for _ in range(2)]
                for scale in [1.0, 2.5]:
                    for opt in [0, 1]:
                        assert lib.check_route(addr(x), addr(b), addr(w[opt]), addr(ids[opt]), rows, scale, opt, stream()) == 0
                    exact(ids[0], ids[1]); exact(w[0], w[1])
                    counts['route'] += 1
                if use_bias and pattern == 'random':
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        assert lib.check_route(addr(x), addr(b), addr(w[1]), addr(ids[1]), rows, 1.0, 1, stream()) == 0
                    for replay in range(5):
                        x.normal_(); b.normal_()
                        if replay == 4:
                            x[:, 1] = float('nan')
                        graph.replay()
                        assert lib.check_route(addr(x), addr(b), addr(w[0]), addr(ids[0]), rows, 1.0, 0, stream()) == 0
                        exact(ids[0], ids[1]); exact(w[0], w[1])
                        counts['graph_replays'] += 1

    for dtype in [torch.float16, torch.bfloat16]:
        for width in [1024, 4096]:
            for rows in [1, 3, 64, 257, 65536]:
                storage = torch.empty(rows * width + 1, device='cuda', dtype=dtype)
                x = storage[1:].view(rows, width)
                x.normal_()
                packed = [torch.empty_like(x, dtype=torch.int8) for _ in range(2)]
                scales = [torch.empty(rows, device='cuda') for _ in range(2)]
                patterns = ['random'] if rows > 257 else ['random', 'zero', 'tiny', 'ties', 'nonfinite']
                for pattern in patterns:
                    if pattern == 'zero':
                        x.zero_()
                    elif pattern == 'tiny':
                        x.fill_(1e-12)
                    elif pattern == 'ties':
                        x.fill_(127); x[:, 0] = 0.5; x[:, 1] = -0.5; x[:, 2] = 1.5
                    elif pattern == 'nonfinite':
                        x.fill_(1); x[:, 0] = float('inf'); x[:, 1] = float('nan')
                    for opt in [0, 1]:
                        assert lib.check_quant(addr(x), addr(packed[opt]), addr(scales[opt]), rows, width, dtype == torch.bfloat16, opt, stream()) == 0
                    exact(packed[0], packed[1]); exact(scales[0], scales[1])
                    counts['quant'] += 1
                if rows == 3:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        assert lib.check_quant(addr(x), addr(packed[1]), addr(scales[1]), rows, width, dtype == torch.bfloat16, 1, stream()) == 0
                    for _ in range(5):
                        x.normal_(); graph.replay()
                        assert lib.check_quant(addr(x), addr(packed[0]), addr(scales[0]), rows, width, dtype == torch.bfloat16, 0, stream()) == 0
                        exact(packed[0], packed[1]); exact(scales[0], scales[1])
                        counts['graph_replays'] += 1

    for dtype_id, dtype in enumerate([torch.float32, torch.float16, torch.bfloat16]):
        for width, n in [(1024, 13), (1024, 4096), (4096, 13), (4096, 1280)]:
            x = torch.randint(-128, 128, (width,), device='cuda', dtype=torch.int8)
            w = torch.randint(-128, 128, (n, width), device='cuda', dtype=torch.int8)
            xs = torch.tensor([0.015], device='cuda')
            ws = torch.rand(n, device='cuda') * 0.01
            dots = (x.cpu().to(torch.int64) * w.cpu().to(torch.int64)).sum(1).float()
            base = dots * (xs.cpu() * ws.cpu())
            out = torch.empty(n, device='cuda', dtype=dtype)
            for bias_stride in [-1, 0, 1, 2]:
                bias = None if bias_stride < 0 else torch.randn(max(n * bias_stride, 1), device='cuda', dtype=dtype)
                expected = base.clone()
                if bias is not None:
                    expected += bias.cpu().float()[torch.arange(n) * bias_stride]
                vector_modes = [0, 1] if width == 4096 else [0]
                for vector in vector_modes:
                    assert lib.check_gemv(addr(out), addr(bias), addr(x), addr(xs), addr(w), addr(ws), n, width, dtype_id, vector, max(bias_stride, 0), stream()) == 0
                    exact(out.cpu(), expected.to(dtype))
                    counts['gemv'] += 1
    torch.cuda.synchronize()
    result = {'passed': True, 'checks': counts, 'library': str(Path(args.library).resolve())}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
