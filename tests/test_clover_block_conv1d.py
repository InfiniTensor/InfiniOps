"""Correctness tests for the native and Triton CloverBlockConv1d providers.

Golden model: per (request, token, channel), silu(left-history 4-tap conv +
right-symmetric 3-tap conv), zeroed on invalid rows. Both providers are checked
against it on identical inputs, covering bf16/fp16, block sizes 1/8/32, dead
slots (-1), per-row validity masks and the read-only state.
"""

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverBlockConv1d"):
    pytest.skip("`CloverBlockConv1d` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10


def torch_golden(x, state, indices, weight, right_weight, valid, block_size):
    tokens, channels = x.shape
    requests = indices.numel()
    out = torch.zeros_like(x, dtype=torch.float32)
    xf, wf, rwf = x.float(), weight.float(), right_weight.float()
    for r in range(requests):
        slot = int(indices[r])
        active_request = 0 <= slot < state.size(0)
        for t in range(block_size):
            token = r * block_size + t
            if not active_request or int(valid[token]) == 0:
                continue
            acc = 0.0
            for tap in range(4):
                rel = t + tap - 3
                if rel >= 0:
                    src = r * block_size + rel
                    value = xf[src] if int(valid[src]) != 0 else 0.0
                else:
                    value = state[slot, :, rel + 3].float()
                acc = acc + value * wf[:, 0, tap]
            for d in range(1, 4):
                right = t + d
                if right < block_size:
                    src = r * block_size + right
                    value = xf[src] if int(valid[src]) != 0 else 0.0
                    acc = acc + value * rwf[:, 0, d - 1]
            out[token] = acc * torch.sigmoid(acc)
    return out


@pytest.mark.parametrize("block_size", (1, 8, 32))
@pytest.mark.parametrize("dtype,rtol", ((torch.bfloat16, 3e-2), (torch.float16, 3e-2)))
@pytest.mark.parametrize("id_dtype", (torch.int32, torch.int64))
def test_block_conv1d(device, block_size, dtype, rtol, id_dtype):
    if device != "cuda":
        pytest.skip("requires a CUDA-like backend")
    impls = infini.ops.CloverBlockConv1d.active_implementation_indices(device)
    if NATIVE not in impls or TRITON not in impls:
        pytest.skip(f"requires the native and Triton providers, active={impls}")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(31)
    requests, channels, slots = 6, 256, 8

    x = torch.randn(requests * block_size, channels, device=device, dtype=dtype)
    state = torch.randn(slots, channels, 3, device=device, dtype=dtype)
    state0 = state.clone()
    # One dead slot and one out-of-range slot exercise the guard paths.
    indices = torch.arange(requests, device=device, dtype=id_dtype) % slots
    indices[4] = -1
    indices[5] = slots + 3
    weight = torch.randn(channels, 1, 4, device=device, dtype=dtype) * 0.2
    right_weight = torch.randn(channels, 1, 3, device=device, dtype=dtype) * 0.2
    valid = torch.ones(requests * block_size, device=device, dtype=torch.int8)
    valid[::5] = 0  # scattered invalid rows inside blocks
    out_n = torch.empty_like(x)
    out_t = torch.empty_like(x)

    infini.ops.clover_block_conv1d(
        x,
        state,
        indices,
        weight,
        right_weight,
        valid,
        block_size,
        out_n,
        stream=stream,
        implementation_index=NATIVE,
    )
    infini.ops.clover_block_conv1d(
        x,
        state,
        indices,
        weight,
        right_weight,
        valid,
        block_size,
        out_t,
        stream=stream,
        implementation_index=TRITON,
    )
    torch.cuda.synchronize()

    golden = torch_golden(
        x, state, indices, weight, right_weight, valid, block_size
    ).to(device)
    torch.testing.assert_close(out_n.float(), golden, rtol=rtol, atol=rtol)
    torch.testing.assert_close(out_n.float(), out_t.float(), rtol=rtol, atol=rtol)
    assert torch.equal(state, state0), "the preview reads the state only"
