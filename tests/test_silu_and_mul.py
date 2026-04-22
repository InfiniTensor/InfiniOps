import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, x_strides, out_strides",
    (
        ((13, 8), None, None),
        ((16, 11264), None, None),
        ((4, 4, 11264), None, None),
        ((1, 8), None, None),
        ((32, 5632), None, None),
        # Non-contiguous `x` (inner stride > inner dim doubled).
        ((13, 8), (16, 1), (4, 1)),
        # Non-contiguous across all dims (3-D with larger outer stride).
        ((4, 4, 16), (128, 16, 1), (64, 8, 1)),
    ),
)
@pytest.mark.parametrize("implementation_index", (0,))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_silu_and_mul(
    shape,
    x_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.SiluAndMul.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    x = rand_strided(shape, x_strides, dtype=dtype, device=device)
    d = shape[-1] // 2
    out_shape = (*shape[:-1], d)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _silu_and_mul(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_silu_and_mul,
        (x, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _silu_and_mul(x, out, implementation_index=0):
    infini.ops.silu_and_mul(
        x,
        -1,
        out,
        implementation_index=implementation_index,
        stream=get_stream(x.device),
    )

    return out


def _torch_silu_and_mul(x, out):
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    result = up * torch.sigmoid(gate) * gate

    return result.to(out.dtype)
