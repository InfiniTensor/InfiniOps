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

_PYTORCH_SLOT = 8


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
@pytest.mark.parametrize("stream_source", ("current", "handle"))
def test_abs_torch_backend_uses_selected_stream(device, stream_source):
    stream_apis = {
        "cuda": (torch.cuda, "cuda_stream"),
        "npu": (getattr(torch, "npu", None), "npu_stream"),
    }
    accelerator, stream_attr = stream_apis.get(device, (None, None))

    if (
        stream_source == "handle"
        and device == "npu"
        and accelerator is not None
        and not hasattr(accelerator, "ExternalStream")
    ):
        pytest.skip("The installed torch_npu does not support external streams")

    if accelerator is None or not hasattr(accelerator, "_sleep"):
        pytest.skip("The device does not expose the required stream test APIs")
    if _PYTORCH_SLOT not in infini.ops.Abs.active_implementation_indices(device):
        pytest.skip(f"PyTorch backend slot 8 is not active on {device}")

    input = torch.full((4096,), -1.0, device=device)
    out = torch.full_like(input, torch.nan)

    def call_abs(**kwargs):
        infini.ops.abs(
            input,
            out,
            implementation_index=_PYTORCH_SLOT,
            **kwargs,
        )

    call_abs()
    out.fill_(torch.nan)
    stream = accelerator.Stream()
    accelerator.synchronize()

    with accelerator.stream(stream):
        accelerator._sleep(1_000_000_000)
        if stream_source == "current":
            call_abs()

    if stream_source == "handle":
        call_abs(stream=getattr(stream, stream_attr))

    accelerator.default_stream().synchronize()
    snapshot = out.clone()
    assert torch.isnan(snapshot).all()

    stream.synchronize()
    torch.testing.assert_close(out, input.abs())
