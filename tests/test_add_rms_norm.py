import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, randn_strided


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
    x1 = randn_strided(shape, strides, dtype=dtype, device=device)
    x2 = randn_strided(shape, strides, dtype=dtype, device=device)
    gamma = randn_strided(weight_shape, None, dtype=dtype, device=device)
    y_out = empty_strided(shape, strides, dtype=dtype, device=device)
    x_out = empty_strided(shape, strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _add_rms_norm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_add_rms_norm,
        (x1, x2, gamma),
        {"eps": eps, "y_out": y_out, "x_out": x_out},
        rtol=rtol,
        atol=atol,
    )


def _add_rms_norm(x1, x2, gamma, *, eps=1e-6, y_out=None, x_out=None,
                  implementation_index=0):
    if x1.device.type == "npu":
        infini.ops.add_rms_norm(
            x1, x2, gamma, eps, y_out, x_out,
            implementation_index=implementation_index,
            stream=get_npu_stream(x1),
        )
    else:
        infini.ops.add_rms_norm(
            x1, x2, gamma, eps, y_out, x_out,
            implementation_index=implementation_index,
        )

    # Concatenate both outputs into a single flat tensor for `allclose` comparison.
    return torch.cat([y_out.contiguous().flatten(), x_out.contiguous().flatten()])


def _torch_add_rms_norm(x1, x2, gamma, *, eps=1e-6, y_out=None, x_out=None):
    x_sum = x1 + x2

    if x_out is not None:
        x_out.copy_(x_sum)

    rms = torch.sqrt(torch.mean(x_sum.float() * x_sum.float(), dim=-1,
                                keepdim=True) + eps)
    y = (x_sum.float() / rms * gamma.float()).to(x1.dtype)

    if y_out is not None:
        y_out.copy_(y)

    return torch.cat([y_out.contiguous().flatten(), x_out.contiguous().flatten()])
