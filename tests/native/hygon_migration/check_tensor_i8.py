"""Validate per-tensor INT8 providers, strided layouts and changing graph inputs."""
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
    lib.check_tensor_i8.argtypes = [ptr] * 4 + [ctypes.c_int] * 2 + [ptr] * 3 + [ctypes.c_int] * 2 + [ptr]
    lib.check_tensor_i8.restype = ctypes.c_int
    torch.manual_seed(20261124)
    counts = {'quant': 0, 'dequant': 0, 'graph_replays': 0, 'invalid': 0, 'public_api': 0}

    def call(x, q, scale, zero=None, dequant=False, static=False, api=False):
        shape = (ctypes.c_size_t * x.ndim)(*x.shape)
        xs = (ctypes.c_ssize_t * x.ndim)(*x.stride())
        qs = (ctypes.c_ssize_t * x.ndim)(*q.stride())
        dtype = [torch.float32, torch.float16, torch.bfloat16].index(x.dtype)
        return lib.check_tensor_i8(x.data_ptr(), q.data_ptr(), scale.data_ptr(),
                                  None if zero is None else zero.data_ptr(), dtype,
                                  x.ndim, ctypes.cast(shape, ptr), ctypes.cast(xs, ptr),
                                  ctypes.cast(qs, ptr), int(dequant) + 2 * api, static,
                                  torch.cuda.current_stream().cuda_stream)

    def exact(actual, expected, case):
        actual = actual.cpu().contiguous()
        expected = expected.contiguous()
        if actual.dtype == torch.int8:
            assert torch.equal(actual, expected), case
            return
        assert torch.equal(torch.isnan(actual), torch.isnan(expected)), case
        mask = ~torch.isnan(actual)
        bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
        assert torch.equal(actual.view(bits)[mask], expected.view(bits)[mask]), case

    def reference(x, static_scale=None):
        x = x.cpu().float()
        if static_scale is None:
            absmax = torch.nan_to_num(x.abs(), nan=0., posinf=float('inf')).amax() if x.numel() else torch.tensor(0.)
            scale = absmax.clamp_min(1.e-10) / 127.
        else:
            scale = static_scale.cpu().reshape(())
        inverse = 1. / scale
        normalized = x * inverse if torch.isfinite(inverse) else x / scale
        q = torch.nan_to_num(normalized, nan=-127., posinf=127., neginf=-127.).clamp(-127, 127).round().to(torch.int8)
        return q, scale.reshape(1)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        layouts = [torch.empty(n, device='cuda', dtype=dtype) for n in [0, 1, 63, 257, 4096, 4097, 16387]]
        layouts += [torch.empty(9, 34, device='cuda', dtype=dtype)[:, ::2].T,
                    torch.empty(2, 3, 514, device='cuda', dtype=dtype)[:, :, ::2]]
        for x in layouts:
            q_store = torch.full((x.numel() * 2 + 1,), -111, dtype=torch.int8, device='cuda')
            q = q_store[1::2].view(x.shape)
            scale = torch.empty(1, device='cuda')
            for pattern in ['random', 'zero', 'ties', 'tiny', 'nonfinite']:
                if pattern == 'random': x.normal_()
                elif pattern == 'zero': x.zero_()
                elif pattern == 'ties':
                    x.fill_(127.)
                    if x.numel():
                        values = torch.tensor([.5, -.5, 1.5, -1.5], device='cuda', dtype=dtype)
                        x.copy_(values.repeat((x.numel() + 3) // 4)[:x.numel()].reshape(x.shape))
                elif pattern == 'tiny': x.fill_(1.e-12)
                else:
                    x.fill_(1.)
                    if x.numel():
                        values = torch.tensor([float('inf'), float('nan'), -float('inf'), 0.], device='cuda', dtype=dtype)
                        x.copy_(values.repeat((x.numel() + 3) // 4)[:x.numel()].reshape(x.shape))
                for static in [False, True]:
                    scale.fill_(1. if pattern == 'ties' else .015)
                    expected, expected_scale = reference(x, scale if static else None)
                    case = (str(dtype), list(x.shape), pattern, static)
                    assert call(x, q, scale, static=static) == 0, case
                    exact(q, expected, case)
                    exact(scale, expected_scale, case)
                    assert torch.all(q_store[::2] == -111), ('padding', case)
                    counts['quant'] += 1
            for scalar in [0., -0., .015, -2., 1.e-8, float('inf'), float('nan')]:
                q.random_(-128, 128)
                scale.fill_(scalar)
                for with_zero in [False, True]:
                    zero = torch.tensor([3.5], device='cuda') if with_zero else None
                    qf = q.cpu().float()
                    expected = ((qf - zero.cpu()[0] if zero is not None else qf) * scale.cpu()[0]).to(dtype)
                    assert call(x, q, scale, zero, dequant=True) == 0
                    exact(x, expected, ('dequant', str(dtype), list(x.shape), scalar, with_zero))
                    counts['dequant'] += 1

        for count in [4096, 4097]:
            x = torch.empty(count, device='cuda', dtype=dtype)
            q = torch.empty_like(x, dtype=torch.int8)
            scale = torch.ones(1, device='cuda')
            for static in [False, True]:
                x.normal_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    assert call(x, q, scale, static=static) == 0
                for magnitude in [100., .01, 0., 2., .001]:
                    x.normal_().mul_(magnitude)
                    scale.fill_(.02 if static else 999.)
                    expected, expected_scale = reference(x, scale if static else None)
                    graph.replay()
                    exact(q, expected, ('graph', count, static, magnitude))
                    exact(scale, expected_scale, ('graph-scale', count, static, magnitude))
                    counts['graph_replays'] += 1

        zero = torch.zeros(1, device='cuda')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert call(x, q, scale, zero, dequant=True) == 0
        for value in [0., .01, -.5, 2., 1.e-8]:
            q.random_(-128, 128)
            scale.fill_(value)
            zero.fill_(value * 3.)
            expected = ((q.cpu().float() - zero.cpu()[0]) * scale.cpu()[0]).to(dtype)
            graph.replay()
            exact(x, expected, ('dequant-graph', str(dtype), value))
            counts['graph_replays'] += 1

        x = torch.randn(9, 34, device='cuda', dtype=dtype)[:, ::2].T
        q = torch.empty_like(x, dtype=torch.int8)
        scale = torch.ones(1, device='cuda')
        for static in [False, True]:
            expected, expected_scale = reference(x, scale if static else None)
            assert call(x, q, scale, static=static, api=True) == 0
            exact(q, expected, ('public-quant', str(dtype), static))
            exact(scale, expected_scale, ('public-scale', str(dtype), static))
            restored = torch.empty_like(x)
            assert call(restored, q, scale, dequant=True, api=True) == 0
            exact(restored, (q.cpu().float() * scale.cpu()[0]).to(dtype), ('public-dequant', str(dtype), static))
            counts['public_api'] += 2

        x = torch.ones(8, device='cuda', dtype=dtype)
        q = torch.empty(8, device='cuda', dtype=torch.int8)
        scale = torch.ones(1, device='cuda')
        zero = torch.zeros_like(scale)
        assert call(x, q, scale, zero) == 1
        assert call(x, q[:1].expand(8), scale) == 1
        assert call(x[:1].expand(8), q, scale, dequant=True) == 1
        counts['invalid'] += 3
    torch.cuda.synchronize()
    result = {'passed': True, 'checks': counts}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
