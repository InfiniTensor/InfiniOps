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
        ((16, 1023), None, None),
        ((16, 1024), None, None),
        ((16, 1025), None, None),
        ((16, 18944), None, None),
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


@pytest.mark.parametrize("padded", (False, True))
def test_silu_and_mul_repeated_small_shape(device, implementation_index, padded):
    if device != "npu" or implementation_index != 0:
        pytest.skip("Ascend native small-shape regression")

    for value in (0.25, 0.75):
        input = torch.empty_strided(
            (13, 8),
            (12, 1) if padded else (8, 1),
            dtype=torch.float16,
            device=device,
        ).fill_(value)
        out = torch.empty_strided(
            (13, 4),
            (6, 1) if padded else (4, 1),
            dtype=torch.float16,
            device=device,
        ).fill_(-9)
        expected = _torch_silu_and_mul(input, torch.empty_like(out)).cpu()
        actual = _silu_and_mul(input, out, implementation_index).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


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
