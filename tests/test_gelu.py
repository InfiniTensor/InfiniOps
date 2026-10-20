import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((), None, None, False),
    ((0,), None, None, False),
    ((13, 4), None, None, False),
    ((13, 4), None, None, True),
    ((13, 4), (10, 1), (10, 1), False),
    ((13, 4), (10, 1), (10, 1), True),
    ((4, 4, 5632), None, None, False),
)

_FLOAT_DTYPE_CASES = (
    (torch.float64, 1e-6, 1e-6),
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides, inplace", _SHAPE_CASES)
@pytest.mark.parametrize("approximate", ("none", "tanh"))
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_gelu(
    shape,
    input_strides,
    out_strides,
    inplace,
    approximate,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 GELU")
    if device == "mlu" and dtype == torch.float64:
        pytest.skip("Cambricon CNNL does not support float64 GELU")

    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        lambda *args: _gelu(*args, implementation_index=implementation_index),
        _torch_gelu,
        (input, approximate, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _gelu(input, approximate, out, *, implementation_index=0):
    infini.ops.gelu(
        input,
        approximate,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_gelu(input, approximate, out):
    out.copy_(torch.nn.functional.gelu(input, approximate=approximate))

    return out


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("shape", [(), (0,), (3, 7), (2, 17, 32)])
def test_encoder_gelu_out_preserves_storage_and_matches_native(
    device, shape, approximate
):
    if device == "npu":
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::gelu", "PrivateUse1"
        )
    x = torch.randn(shape, device=device, dtype=torch.float32)
    storage = torch.full((*shape, 2), -123.0, device=device)
    out = storage[..., 0]
    address = out.data_ptr()
    infini.ops.gelu(
        x, approximate, out, implementation_index=8, stream=get_stream(x.device)
    )
    expected = torch.nn.functional.gelu(x, approximate=approximate)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    assert out.data_ptr() == address
    assert torch.all(storage[..., 1] == -123.0)


def test_external_stream_operations_are_visible_to_following_copies(device):
    x = torch.linspace(-2.0, 2.0, 119, device=device).reshape(7, 17)
    out = torch.empty_like(x)
    for _ in range(12):
        x.add_(0.01)
        infini.ops.exp(x, out, implementation_index=8, stream=get_stream(x.device))
        # A following device copy must see the just-submitted operation.
        copied = out.clone()
        torch.testing.assert_close(copied, x.exp(), atol=1e-5, rtol=1e-5)
