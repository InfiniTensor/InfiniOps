"""Check public Iluvatar operators through the locally built C++ probe."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
import r5_operator_probe as probe

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
a = p.parse_args()
torch.manual_seed(42)
records = []
stream = torch.cuda.current_stream().cuda_stream


def check(name, x, y, other=None, inplace=False):
    original = x.clone()
    if inplace:
        y = x
    probe.run(name, x, other, y, stream)
    torch.cuda.synchronize()
    if name == "softmax":
        expected = torch.softmax(original.float(), dim=-1).to(x.dtype)
        tolerance = (
            0.01
            if x.dtype == torch.bfloat16
            else 0.003
            if x.dtype == torch.float16
            else 1e-5
        )
    elif name == "swiglu":
        gate, up = original.chunk(2, dim=-1)
        expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
        tolerance = (
            0.01
            if x.dtype == torch.bfloat16
            else 0.001
            if x.dtype == torch.float16
            else 1e-6
        )
    else:
        expected = original + other
        tolerance = 0
    torch.testing.assert_close(y, expected, atol=tolerance, rtol=tolerance)
    raw = y.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    records.append(
        dict(
            operation=name,
            shape=list(x.shape),
            dtype=str(x.dtype),
            strides=list(x.stride()),
            inplace=inplace,
            max_abs=float((y.float() - expected.float()).abs().max()),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    )
    print("PASS", name, tuple(x.shape), str(x.dtype), inplace, flush=True)


for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for height, width in (
        (1, 1),
        (17, 65),
        (17, 127),
        (17, 128),
        (17, 129),
        (255, 255),
        (256, 256),
        (257, 257),
        (511, 512),
        (512, 512),
        (513, 513),
        (1023, 1024),
        (1025, 1025),
        (2047, 2048),
        (2048, 2048),
        (16, 4096),
    ):
        x = torch.randn((2, height, width), device="cuda", dtype=dtype)
        check("softmax", x, torch.empty_like(x))
        if dtype == torch.bfloat16:
            check("softmax", x.clone(), None, inplace=True)
            mask = (
                torch.arange(width, device="cuda")[None, :]
                > torch.arange(height, device="cuda")[:, None] + width - height
            )
            check("softmax", x.masked_fill(mask, float("-inf")), None, inplace=True)
    x = torch.randn((2, 17, 128), device="cuda", dtype=dtype)[:, :, :65]
    check("softmax", x, torch.empty_like(x))
    for width in (1, 17, 1023, 1024, 1025, 5632, 18944):
        x = torch.randn((16, width * 2), device="cuda", dtype=dtype)
        check("swiglu", x, torch.empty((16, width), device="cuda", dtype=dtype))
    x = torch.randn((3, 17 * 2 + 8), device="cuda", dtype=dtype)[:, :34]
    check("swiglu", x, torch.empty((3, 17), device="cuda", dtype=dtype))
    for shape in ((16, 3584), (4, 17, 3584)):
        x = torch.randn(shape, device="cuda", dtype=dtype)
        other = torch.randn_like(x)
        check("add", x, torch.empty_like(x), other)
        check("add", x.clone(), None, other, inplace=True)
        check("add", x, torch.empty_like(x), other[0:1])
for tokens in (1, 4, 17):
    for packed in (False, True):
        for neox in (False, True):
            for inverse in (False, True):
                dtype = torch.bfloat16
                if packed:
                    storage = torch.randn(
                        (tokens, 3, 28, 128), device="cuda", dtype=dtype
                    )
                    q, k = storage[:, 0], storage[:, 1]
                    untouched = storage[:, 2].clone()
                else:
                    q = torch.randn((tokens, 28, 128), device="cuda", dtype=dtype)
                    k = torch.randn_like(q)
                positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
                cache = torch.randn((tokens, 128), device="cuda", dtype=dtype)

                def expected(value):
                    x = value.float().clone()
                    cos, sin = cache.float().chunk(2, -1)
                    cos = cos[:, None, :]
                    sin = sin[:, None, :] * (-1 if inverse else 1)
                    first, second = (
                        (x[..., :64], x[..., 64:])
                        if neox
                        else (x[..., ::2], x[..., 1::2])
                    )
                    result = x.clone()
                    if neox:
                        result[..., :64] = first * cos - second * sin
                        result[..., 64:] = second * cos + first * sin
                    else:
                        result[..., ::2] = first * cos - second * sin
                        result[..., 1::2] = second * cos + first * sin
                    return result.to(dtype)

                q_ref, k_ref = expected(q), expected(k)
                probe.rope(q, k, positions, cache, neox, inverse, stream)
                torch.cuda.synchronize()
                torch.testing.assert_close(q, q_ref, rtol=0.02, atol=0.01)
                torch.testing.assert_close(k, k_ref, rtol=0.02, atol=0.01)
                if packed:
                    assert torch.equal(storage[:, 2], untouched)
                raw = b"".join(
                    t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    for t in (q, k)
                )
                records.append(
                    dict(
                        operation="rope",
                        tokens=tokens,
                        packed=packed,
                        neox=neox,
                        inverse=inverse,
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
                print("PASS rope", tokens, packed, neox, inverse, flush=True)
Path(a.output).write_text(json.dumps(records, indent=2) + "\n")
print("OPERATOR_CHECK_COMPLETE", len(records), flush=True)
