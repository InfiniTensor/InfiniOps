"""Strict Hygon W8A8 regression and optional paired performance screen.

The baseline retains the pre-optimization scalar GPU kernels. All outputs and
workspace bytes must match. Input quantization is independently checked on CPU,
using direct division with one FP32 rounding instead of reciprocal-multiply.
Performance against this baseline is not external indicator-10 acceptance.
"""
from pathlib import Path
import argparse
import ctypes
import json
import statistics
import torch

def main():
    parser = argparse.ArgumentParser(
        description='Check Hygon W8A8 expert SDK calls against the preserved scalar kernels')
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', default='verification-experts')
    parser.add_argument('--performance', action='store_true')
    options = parser.parse_args()
    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=True)
    lib = ctypes.CDLL(options.library)
    fn = lib.run
    ptr = ctypes.c_void_p
    fn.argtypes = [
        ptr] * 9 + [ctypes.c_size_t] * 5 + [ctypes.c_float,
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ptr]
    fn.restype = ctypes.c_int
    torch.manual_seed(20261125)
    checks = 0
    graphs = 0
    quant_checks = 0

    def inputs(m, h, inner, e, k, dtype):
        x = torch.randn((m, h), device='cuda', dtype=dtype)
        weights = torch.rand((m, k), device='cuda')
        ids = torch.randint(0, e, (m, k), device='cuda', dtype=torch.int64)
        g = torch.randint(-128, 128, (e, 2 * inner, h), device='cuda', dtype=torch.int8)
        gs = torch.rand((e, 2 * inner, 1), device='cuda') * 0.01
        d = torch.randint(-128, 128, (e, h, inner), device='cuda', dtype=torch.int8)
        ds = torch.rand((e, h, 1), device='cuda') * 0.01
        count = (m * k * inner + m + m * k) * 4 + m * h + m * k * inner
        storage = [
            torch.full((count + 64,), 97, device='cuda', dtype=torch.uint8) for _ in range(2)]
        ws = [z[32:-32] for z in storage]
        ys = [torch.empty_like(x) for _ in range(2)]
        return (x, weights, ids, g, gs, d, ds, storage, ws, ys)

    def runner(args, mode, routed, limit, scale):
        (x, weights, ids, g, gs, d, ds, storage, ws, ys) = args
        (m, h) = x.shape
        (e, _, inner) = d.shape
        k = ids.shape[1]

        def run():
            status = fn(
                x.data_ptr(), weights.data_ptr(), ids.data_ptr(), g.data_ptr(),
                gs.data_ptr(), d.data_ptr(), ds.data_ptr(), ys[mode].data_ptr(),
                ws[mode].data_ptr(), m, h, inner, e, k, limit, routed,
                scale, mode, int(x.dtype == torch.bfloat16), torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError('expert launch failed ' + str(status))
        return run

    def verify(args, context):
        nonlocal checks, quant_checks
        (x, weights, ids, g, gs, d, ds, storage, ws, ys) = args
        (m, h) = x.shape
        inner = d.shape[2]
        routes = ids.numel()
        torch.cuda.synchronize()
        if not torch.equal(
            ys[0].view(torch.uint8), ys[1].view(torch.uint8)) or not torch.equal(
            ws[0], ws[1]):
            (out / 'failure.json').write_text(
                json.dumps({'context': context, 'output_equal': torch.equal(ys[0].view(torch.uint8),
                ys[1].view(torch.uint8)), 'workspace_equal': torch.equal(ws[0],
                ws[1]), 'max_abs': float((ys[0].float() - ys[1].float()).abs().max()),
                'workspace_different_bytes': int((ws[0] != ws[1]).sum())}))
            raise RuntimeError('Expert output or intermediate mismatch')
        for z in storage:
            assert bool((z[:32] == 97).all() and (z[-32:] == 97).all())
        cpu = x.cpu().float()
        maximum = cpu.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        scale = (maximum.double() / 127).float()
        inverse = (torch.full_like(maximum.double(), 127) / maximum.double()).float()
        expected = (cpu * inverse).round().clamp(-127, 127).to(torch.int8)
        offset = (routes * inner + m + routes) * 4
        actual_q = ws[1][offset:offset + m * h].view(torch.int8).reshape(m, h).cpu()
        actual_s = ws[1][routes * inner * 4:(routes * inner + m) * 4].view(torch.float32).cpu()
        assert torch.equal(
            actual_q, expected) and torch.equal(actual_s.view(torch.uint8),
            scale.flatten().view(torch.uint8))
        checks += 1
        quant_checks += 1
    for dtype in [torch.float32, torch.bfloat16]:
        for (m, h, inner) in [(1, 256, 64), (3, 1024, 128), (17, 256, 64), (2, 259, 67)]:
            for (routed, k, limit) in [(False, 1, 7.0), (True, 6, 1.0), (True, 6, 7.0)]:
                args = inputs(m, h, inner, 8, k, dtype)
                args[2].view(-1)[0] = -1
                if args[2].numel() > 1:
                    args[2].view(-1)[1] = 8
                runs = [runner(args, i, routed, limit, 1.5 if routed else 1.0) for i in [0, 1]]
                for run in runs:
                    run()
                verify(
                    args, {'shape': [m, h, inner], 'dtype': str(dtype),
                    'routed': routed, 'k': k, 'phase': 'eager'})
                gs = []
                for run in runs:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    gs.append(graph)
                for seed in range(3):
                    args[0].fill_(0.0 if seed == 0 else 1.0 if seed == 1 else -64.0)
                    args[2].fill_(seed % 8)
                    for graph in gs:
                        graph.replay()
                    verify(
                        args, {'shape': [m, h, inner], 'dtype': str(dtype),
                        'routed': routed, 'k': k, 'phase': 'graph', 'seed': seed})
                    graphs += 1
    cases = []
    for dtype in [torch.float32, torch.bfloat16]:
        for (m,
            h, inner, k) in [(1, 4, 4, 1), (2, 4, 8, 6), (3, 68,
            12, 6), (16, 64, 64, 1), (17, 64, 64, 1), (257, 64,
            256, 6), (257, 1024, 4, 1), (65536, 4, 4, 1), (1,
            132104, 4, 1)]:
            for misaligned in [False, True] if m < 100 and h < 100 else [False]:
                args = list(inputs(m, h, inner, 1 if k == 1 else 8, k, dtype))
                if misaligned:
                    for index in [3, 5]:
                        old = args[index]
                        storage = torch.empty(old.numel() + 1, device='cuda', dtype=torch.int8)
                        shift = storage[1:].reshape(old.shape)
                        shift.copy_(old)
                        args[index] = shift
                if h == 132104:
                    args[0].fill_(127)
                    args[3].fill_(-128)
                    args[5].fill_(127)
                elif m < 100:
                    args[0].zero_()
                    args[0][:, 0] = 127
                    args[0][:, 1] = 0.5
                    args[0][:, 2] = -1.5
                    args[0][:, 3] = 2.5
                args[2].zero_()
                args[1].fill_(0.25)
                routed = k > 1
                runs = [runner(args, i, routed, 7.0, 0.25 if routed else 1.0) for i in [0, 1]]
                for run in runs:
                    run()
                context = {
                    'shape': [m, h, inner], 'dtype': str(dtype), 'misaligned_weights': misaligned,
                    'routed': routed}
                verify(args, context)
                cases.append(context)
                if m < 100 and h < 100:
                    gs = []
                    for run in runs:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            run()
                        gs.append(g)
                    for value in [1e-12, -1e-12]:
                        args[0].fill_(value)
                        for g in gs:
                            g.replay()
                        verify(args, dict(context, tiny_input=value))
                        graphs += 1
    (out / 'correctness.json').write_text(
        json.dumps({'passed': True, 'checks': checks, 'graph_checks': graphs,
        'independent_input_quant_checks': quant_checks, 'intermediates_bitwise_equal': True,
        'guard_canaries': True}))
    if not options.performance:
        print(json.dumps({'passed': True,
            'checks': checks, 'graph_checks': graphs, 'edge_cases': len(cases)}))
        return
    results = []
    for dtype in [torch.float32, torch.bfloat16]:
        for (m,
            h, inner, e, k, routed) in [(1, 4096, 512, 16, 6,
            True), (8, 4096, 512, 16, 6, True), (32, 1024, 256,
            8, 6, True), (17, 1024, 512, 1, 1, False)]:
            args = inputs(m, h, inner, e, k, dtype)
            runs = [runner(args, i, routed, 7.0, 1.5 if routed else 1.0) for i in [0, 1]]
            gs = []
            for run in runs:
                for _ in range(10):
                    run()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run()
                gs.append(g)
            for graph in gs:
                graph.replay()
            verify(args, {'shape': [m, h, inner], 'dtype': str(dtype), 'phase': 'performance'})
            rounds = []
            for repeat in range(5):
                values = [[], []]
                for sample in range(20):
                    for mode in [0, 1] if (repeat + sample) % 2 == 0 else [1, 0]:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        gs[mode].replay()
                        end.record()
                        end.synchronize()
                        values[mode].append(start.elapsed_time(end) * 1000)
                base = statistics.mean(values[0])
                opt = statistics.mean(values[1])
                rounds.append(
                    {'baseline_us': base, 'candidate_us': opt, 'throughput_gain': 100 * (base / opt - 1),
                    'baseline_samples_us': values[0], 'candidate_samples_us': values[1]})
            results.append(
                {'shape': [m, h, inner], 'dtype': str(dtype), 'routed': routed,
                'rounds': rounds, 'min_gain': min((x['throughput_gain'] for x in rounds))})
            (out / 'performance.json').write_text(
                json.dumps({'baseline': 'unchanged new repository expert kernels',
                'candidate': 'public SDK wave64 integer-dot path',
                'formal_acceptance': False, 'results': results}, indent=2))
    print(json.dumps({'passed': True,
        'checks': checks, 'graph_checks': graphs, 'performance_cases': len(results)}))
if __name__ == '__main__':
    main()
