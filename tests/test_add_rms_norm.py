import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, strides",
    (
        ((1, 64), None),
        ((2, 128), None),
        ((4, 48, 64), None),
        ((2, 4, 2048), None),
        ((1, 64), (64, 1)),
        ((4, 48, 64), (3072, 64, 1)),
    ),
)
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_add_rms_norm(
    shape,
    strides,
    eps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.AddRmsNorm.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    weight_shape = (shape[-1],)
    input = randn_strided(shape, strides, dtype=dtype, device=device)
    residual = randn_strided(shape, strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, strides, dtype=dtype, device=device)
    residual_out = empty_strided(shape, strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _add_rms_norm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_add_rms_norm,
        (input, residual, weight),
        {"eps": eps, "out": out, "residual_out": residual_out},
        rtol=rtol,
        atol=atol,
    )


def _add_rms_norm(
    input,
    residual,
    weight,
    *,
    eps=1e-6,
    out=None,
    residual_out=None,
    implementation_index=0,
):
    infini.ops.add_rms_norm(
        input,
        residual,
        weight,
        eps,
        out,
        residual_out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    # Concatenate both outputs into a single flat tensor for `allclose` comparison.
    return torch.cat([out.contiguous().flatten(), residual_out.contiguous().flatten()])


def _torch_add_rms_norm(input, residual, weight, *, eps=1e-6, out=None, residual_out=None):
    x_sum = input + residual

    if residual_out is not None:
        residual_out.copy_(x_sum)

    rms = torch.sqrt(
        torch.mean(x_sum.float() * x_sum.float(), dim=-1, keepdim=True) + eps
    )
    y = (x_sum.float() / rms * weight.float()).to(input.dtype)

    if out is not None:
        out.copy_(y)

    return torch.cat([out.contiguous().flatten(), residual_out.contiguous().flatten()])
