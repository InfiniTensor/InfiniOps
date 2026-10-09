import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "out_shape, input_strides, out_strides",
    (
        ((13, 4), None, None),
        ((13, 4), (12, 1), (6, 1)),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (48, 10, 1), (24, 6, 1)),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (6656, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (53248, 13312, 1), (26624, 6656, 1)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_silu_and_mul(
    out_shape,
    input_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input_shape = (*out_shape[:-1], out_shape[-1] * 2)
    input = rand_strided(input_shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _silu_and_mul(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_silu_and_mul,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def test_silu_and_mul_non_default_stream(device, implementation_index):
    if device == "cuda":
        accelerator = torch.cuda
        stream_attribute = "cuda_stream"
    elif device == "musa":
        accelerator = torch.musa
        stream_attribute = "musa_stream"
    elif device == "mlu":
        accelerator = torch.mlu
        stream_attribute = "mlu_stream"
    else:
        pytest.skip("non-default streams require an accelerator backend")

    input = torch.randn((32, 128), dtype=torch.float16, device=device)
    out = torch.zeros((32, 64), dtype=torch.float16, device=device)
    expected = _torch_silu_and_mul(input, torch.empty_like(out)).cpu()
    accelerator.synchronize()

    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())
    stream_ptr = getattr(stream, stream_attribute)

    # Keep the current stream busy. A provider call that ignores the explicit
    # stream will be queued behind this delay, while the requested stream can
    # complete independently.
    accelerator._sleep(50_000_000)
    try:
        infini.ops.silu_and_mul(
            input,
            out,
            implementation_index=implementation_index,
            stream=stream_ptr,
        )
        stream.synchronize()
        with accelerator.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    finally:
        accelerator.synchronize()


def _silu_and_mul(input, out, implementation_index=0):
    infini.ops.silu_and_mul(
        input,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_silu_and_mul(input, out):
    hidden_size = input.shape[-1] // 2
    gate = input[..., :hidden_size]
    up = input[..., hidden_size:]

    return torch.mul(torch.nn.functional.silu(gate), up, out=out)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("tokens", (1, 31, 8192))
@pytest.mark.parametrize("hidden", (3072, 6144, 8192))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_silu_and_mul_ascend_token_tiles(
    tokens, hidden, padded, dtype, device, implementation_index
):
    if device != "npu" or implementation_index != 0:
        pytest.skip("token tile regression targets native Ascend")

    input = rand_strided(
        (tokens, hidden * 2),
        (hidden * 2 + 128, 1) if padded else None,
        dtype=dtype,
        device=device,
    )
    out = empty_strided(
        (tokens, hidden),
        (hidden + 128, 1) if padded else None,
        dtype=dtype,
        device=device,
    )

    return Payload(
        lambda *args: _silu_and_mul(*args, implementation_index=implementation_index),
        _torch_silu_and_mul,
        (input, out),
        {},
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
        atol=5e-3 if dtype == torch.bfloat16 else 1e-5,
    )
