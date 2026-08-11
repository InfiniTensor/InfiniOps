import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((4, 4, 5632), None, None),
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 5e-3),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides", _SHAPE_CASES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_abs(
    shape,
    input_strides,
    out_strides,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda input, out: _abs(input, out, implementation_index),
        _torch_abs,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _abs(input, out, implementation_index):
    infini.ops.abs(
        input,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_abs(input, out):
    out.copy_(torch.abs(input))

    return out


@pytest.mark.smoke
def test_abs_torch_backend_uses_handle_stream(device):
    pytorch_slot = 8

    if device != "cuda":
        pytest.skip("The stream regression requires CUDA")
    if not infini.ops.Abs.active_implementation_indices("nvidia"):
        pytest.skip("The stream regression requires the NVIDIA backend")
    if pytorch_slot not in infini.ops.Abs.active_implementation_indices(device):
        pytest.skip("The PyTorch backend is not active")

    input = torch.full((4096,), -1.0, device=device)
    out = torch.full_like(input, torch.nan)
    stream = torch.cuda.Stream()

    def call_abs():
        infini.ops.abs(
            input,
            out,
            stream=stream.cuda_stream,
            implementation_index=pytorch_slot,
        )

    try:
        call_abs()
        stream.synchronize()
        out.fill_(torch.nan)
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
        call_abs()

        default_stream = torch.cuda.default_stream()
        with torch.cuda.stream(default_stream):
            snapshot = out.clone()
        default_stream.synchronize()
        assert torch.isnan(snapshot).all()

        stream.synchronize()
        torch.testing.assert_close(out, input.abs())
    finally:
        torch.cuda.synchronize()
