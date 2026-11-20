"""Validate Clover preview outputs and read-only clean-state behavior."""

import infini.ops as ops
import pytest
import torch


@pytest.fixture(autouse=True)
def _ieee_reference():
    previous = torch.backends.fp32_precision
    torch.backends.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.fp32_precision = previous


def test_clover_preview_reference_and_state():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires a Hygon Torch environment")
    if not hasattr(
        ops, "CloverLocalMemory"
    ) or not ops.CloverLocalMemory.active_implementation_indices("hygon"):
        pytest.skip("requires Hygon Triton Clover providers")
    torch.manual_seed(17)
    stream = torch.cuda.current_stream().cuda_stream
    for width in (1, 8, 32):
        for batch in (1, 3):
            dtype = torch.bfloat16
            channels = 128
            heads = 4
            x = torch.randn(batch * width, channels, device="cuda", dtype=dtype)
            state = torch.randn(batch + 1, channels, 3, device="cuda", dtype=dtype)
            state0 = state.clone()
            indices = torch.arange(batch, device="cuda", dtype=torch.int32)
            weight = torch.randn(channels, 1, 4, device="cuda", dtype=dtype)
            right = torch.randn(channels, 1, 3, device="cuda", dtype=dtype)
            valid = torch.ones(batch * width, device="cuda", dtype=torch.int8)
            out = torch.empty_like(x)
            ops.clover_block_conv1d(
                x, state, indices, weight, right, valid, width, out, stream=stream
            )
            ref = []
            for b in range(batch):
                row = x.view(batch, width, channels)[b]
                joined = torch.cat((state[b].T, row), 0).float()
                y = torch.zeros_like(row, dtype=torch.float32)
                for t in range(width):
                    y[t] = (joined[t : t + 4] * weight[:, 0, :].T.float()).sum(0)
                    for j in range(1, 4):
                        if t + j < width:
                            y[t] += row[t + j].float() * right[:, 0, j - 1].float()
                ref.append(torch.nn.functional.silu(y).to(dtype))
            torch.testing.assert_close(
                out, torch.stack(ref).view_as(out), rtol=0.02, atol=0.04
            )
            assert torch.equal(state, state0)
            q = torch.randn(1, batch * width, 2, 128, device="cuda", dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn(1, batch * width, heads, 128, device="cuda", dtype=dtype)
            g = -torch.rand(1, batch * width, heads, device="cuda")
            s = torch.randn(batch + 1, heads, 128, 128, device="cuda")
            s0 = s.clone()
            scale = torch.full((heads,), 10.0, device="cuda")
            bias = torch.full_like(scale, 5.0)
            hc = torch.ones_like(scale)
            y = torch.empty_like(v)
            ops.clover_local_memory(
                q, k, v, g, s, indices, valid, scale, bias, hc, width, y, stream=stream
            )
            qn = (
                (
                    q.float()
                    * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
                )
                .to(dtype)
                .float()
            )
            kn = (
                (
                    k.float()
                    * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
                )
                .to(dtype)
                .float()
            )
            refs = []
            for b in range(batch):
                qi = (
                    qn[0, b * width : (b + 1) * width]
                    .repeat_interleave(2, 1)
                    .permute(1, 0, 2)
                )
                ki = (
                    kn[0, b * width : (b + 1) * width]
                    .repeat_interleave(2, 1)
                    .permute(1, 2, 0)
                )
                vi = v[0, b * width : (b + 1) * width].permute(1, 0, 2).float()
                scores = qi @ ki * 10
                history_logit = 5 + torch.log(torch.tensor(width / 32.0))
                weights = torch.softmax(
                    torch.cat(
                        (
                            scores,
                            torch.full((heads, width, 1), history_logit, device="cuda"),
                        ),
                        -1,
                    ),
                    -1,
                )
                history = qi @ s[b].to(dtype).float().transpose(-1, -2)
                local = weights[:, :, :width].to(dtype).float() @ vi
                oi = (
                    history
                    * (
                        g[0, b * width : (b + 1) * width].T.exp()
                        * weights[:, :, -1]
                        / 128**0.5
                    ).unsqueeze(-1)
                    + local
                )
                refs.append(oi.permute(1, 0, 2))
            ref = torch.cat(refs).unsqueeze(0).to(dtype)
            torch.testing.assert_close(y, ref, rtol=0.035, atol=0.04)
            assert torch.equal(s, s0)
