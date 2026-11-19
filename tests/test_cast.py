from contextlib import nullcontext

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4, 4), None, None),
        ((16, 5632), None, None),
        ((4, 4, 5632), None, None),
    ),
)
@pytest.mark.parametrize(
    ("input_dtype", "out_dtype", "rtol", "atol"),
    (
        (torch.float16, torch.float32, 1e-3, 1e-3),
        (torch.float32, torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, torch.float32, 1e-2, 5e-3),
        (torch.float32, torch.bfloat16, 1e-2, 5e-3),
        (torch.float16, torch.bfloat16, 1e-2, 5e-3),
        (torch.bfloat16, torch.float16, 1e-2, 5e-3),
    ),
)
def test_cast(
    shape,
    input_strides,
    out_strides,
    input_dtype,
    out_dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=input_dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=out_dtype, device=device)
    cast = _cast

    if (
        device == "cuda"
        and infini.ops.Cast.active_implementation_indices("nvidia")
        and (not input.is_contiguous() or not out.is_contiguous())
    ):
        if 8 not in infini.ops.Cast.active_implementation_indices("nvidia"):
            pytest.skip("strided NVIDIA Cast requires ATen slot 8")

        cast = _cast_aten

    return Payload(
        cast,
        _torch_cast,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cast(input, out, **kwargs):
    infini.ops.cast(input, out, stream=get_stream(input.device), **kwargs)

    return out


def _cast_aten(input, out):
    return _cast(input, out, implementation_index=8)


def _torch_cast(input, out):
    out.copy_(input.to(out.dtype))

    return out


@pytest.mark.smoke
def test_native_cuda_cast_rejects_strided_layout(device):
    if device != "cuda" or 0 not in infini.ops.Cast.active_implementation_indices(
        "nvidia"
    ):
        pytest.skip("native NVIDIA Cast is not active")

    input = randn_strided((13, 4), (10, 1), device=device)
    out = empty_strided((13, 4), (10, 1), dtype=torch.float16, device=device)

    with pytest.raises(ValueError, match="does not support these parameters"):
        infini.ops.cast(input, out, implementation_index=0, stream=get_stream(device))


@pytest.fixture
def require_aten_cast(device):
    if 8 not in infini.ops.Cast.active_implementation_indices(device):
        pytest.skip(f"Cast slot 8 is not active on `{device}`")


@pytest.mark.usefixtures("require_aten_cast")
@pytest.mark.parametrize(
    "input_dtype, out_dtype",
    [
        (torch.int64, torch.float32),
        (torch.int32, torch.float32),
        (torch.bool, torch.float32),
        (torch.bool, torch.int64),
        (torch.int64, torch.int32),
        (torch.float32, torch.int64),
        (torch.float32, torch.float16),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_encoder_cast_preserves_output(input_dtype, out_dtype, strided, device):
    values = torch.tensor([-3.5, -1.0, 0.0, 1.0, 4.5, 17.0], device=device)
    x = values.to(input_dtype).reshape(2, 3)

    if strided:
        source = torch.empty((2, 3, 2), device=device, dtype=input_dtype)
        source[..., 0] = x
        x = source[..., 0]

    backing = torch.full((2, 3, 2), -123, device=device, dtype=out_dtype)
    out = backing[..., 0] if strided else torch.empty_like(x, dtype=out_dtype)
    address = out.data_ptr()
    infini.ops.cast(x, out, implementation_index=8, stream=get_stream(device))
    torch.testing.assert_close(out, x.to(out_dtype), rtol=0, atol=0)
    assert out.data_ptr() == address

    if strided:
        assert torch.all(backing[..., 1] == -123)


@pytest.mark.usefixtures("require_aten_cast")
def test_encoder_cast_scalar_empty_and_same_view(device):
    for shape in [(), (0, 3), (2, 3)]:
        x = torch.ones(shape, device=device, dtype=torch.int64)
        out = torch.empty(shape, device=device, dtype=torch.float32)
        infini.ops.cast(x, out, implementation_index=8, stream=get_stream(device))
        torch.testing.assert_close(out, x.float())
        infini.ops.cast(out, out, implementation_index=8, stream=get_stream(device))
        torch.testing.assert_close(out, x.float())


@pytest.mark.usefixtures("require_aten_cast")
def test_encoder_cast_rejects_shape_change(device):
    x = torch.ones((2, 3), device=device, dtype=torch.int64)
    out = torch.empty((3, 2), device=device)

    with pytest.raises(ValueError, match="does not support these parameters"):
        infini.ops.cast(x, out, implementation_index=8, stream=get_stream(device))


@pytest.mark.usefixtures("require_aten_cast")
def test_encoder_cast_rejects_partial_overlap(device):
    storage = torch.arange(9, device=device, dtype=torch.float32)
    infini.ops.cast(
        storage[:6],
        torch.empty(6, device=device),
        implementation_index=8,
        stream=get_stream(device),
    )

    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        infini.ops.cast(
            storage[:6], storage[1:7], implementation_index=8, stream=get_stream(device)
        )


@pytest.mark.usefixtures("require_aten_cast")
def test_encoder_cast_rejects_internally_overlapping_output(device):
    x = torch.arange(6, device=device, dtype=torch.int64).reshape(2, 3)
    out = torch.empty(4, device=device).as_strided((2, 3), (1, 1))

    with pytest.raises(RuntimeError, match="overlap"):
        infini.ops.cast(x, out, implementation_index=8, stream=get_stream(device))


def test_encoder_cast_default_float_conversion(device):
    x = torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3)
    out = torch.empty_like(x, dtype=torch.float16)
    infini.ops.cast(x, out, stream=get_stream(device))
    torch.testing.assert_close(out, x.half())


@pytest.mark.usefixtures("require_aten_cast")
def test_encoder_cast_nondefault_stream(device):
    stream = torch.cuda.Stream() if device == "cuda" else None

    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        x = torch.arange(12, device=device, dtype=torch.int64).reshape(3, 4)
        out = torch.empty((3, 4), device=device)

    original = get_stream(device)
    infini.ops.cast(
        x,
        out,
        implementation_index=8,
        stream=stream.cuda_stream if stream is not None else original,
    )
    assert get_stream(device) == original
    if stream is not None:
        stream.synchronize()

    torch.testing.assert_close(out, x.float())
