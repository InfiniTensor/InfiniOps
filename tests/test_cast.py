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

    return Payload(
        _cast,
        _torch_cast,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cast(input, out):
    infini.ops.cast(input, out, stream=get_stream(input.device))

    return out


def _torch_cast(input, out):
    out.copy_(input.to(out.dtype))

    return out


@pytest.mark.parametrize(
    "input_dtype,out_dtype",
    [
        (torch.int64, torch.float32),
        (torch.float32, torch.int64),
        (torch.bool, torch.float32),
        (torch.float32, torch.bool),
        (torch.int32, torch.int64),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("shape", [(), (0,), (3, 17), (2, 7, 32)])
def test_encoder_cast_integer_boolean_and_strided_output(
    device, input_dtype, out_dtype, shape
):
    count = torch.tensor(shape).prod().item() if shape else 1
    x = (torch.arange(count, device=device) % 5 - 2).reshape(shape).to(input_dtype)
    storage = torch.full((*shape, 2), 1, device=device, dtype=out_dtype)
    out = storage[..., 0]
    address = out.data_ptr()
    infini.ops.cast(x, out, stream=get_stream(x.device), implementation_index=8)
    torch.testing.assert_close(out, x.to(out_dtype), atol=0, rtol=0)
    assert out.data_ptr() == address
    assert torch.all(storage[..., 1] == 1)


def test_encoder_cast_rechecks_aliases_after_cache_hit(device):
    backing = torch.arange(9, dtype=torch.float32, device=device)
    x, out = backing[:-1], backing[1:]
    infini.ops.cast(
        x, torch.empty_like(x), implementation_index=8, stream=get_stream(x.device)
    )
    before = backing.clone()
    with pytest.raises((ValueError, RuntimeError)):
        infini.ops.cast(x, out, implementation_index=8, stream=get_stream(x.device))
    torch.testing.assert_close(backing, before)
    infini.ops.cast(x, x, implementation_index=8, stream=get_stream(x.device))
    torch.testing.assert_close(backing, before)


def test_encoder_cast_expanded_same_view_is_a_noop(device):
    storage = torch.tensor([3.0], device=device)
    expanded = storage.expand(3, 17)
    infini.ops.cast(
        expanded, expanded, implementation_index=8, stream=get_stream(expanded.device)
    )
    torch.testing.assert_close(storage, torch.tensor([3.0], device=device))
