import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, weight_strides, out_strides",
    (
        ((1, 64), (64,), None, None, None),
        ((2, 128), (128,), None, None, None),
        ((4, 48, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
        ((1, 64), (64,), (64, 1), (1,), (64, 1)),
        ((4, 48, 64), (64,), (3072, 64, 1), (1,), (3072, 64, 1)),
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
def test_rms_norm(
    input_shape,
    weight_shape,
    input_strides,
    weight_strides,
    out_strides,
    eps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.RmsNorm.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(input_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _rms_norm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_rms_norm,
        (input, weight),
        {"eps": eps, "out": out},
        rtol=rtol,
        atol=atol,
    )


def _rms_norm(input, weight, *, eps=1e-6, out=None, implementation_index=0):
    if input.device.type == "npu":
        infini.ops.rms_norm(
            input,
            weight,
            eps,
            out,
            implementation_index=implementation_index,
            stream=get_npu_stream(input),
        )
    else:
        infini.ops.rms_norm(
            input,
            weight,
            eps,
            out,
            implementation_index=implementation_index,
        )

    return out


def _torch_rms_norm(input, weight, *, eps=1e-6, out=None):
    # Fallback for `torch<2.3`: `rms_norm = (x / sqrt(mean(x^2) + eps)) * weight`.
    def _fallback(input, _normalized_shape, weight, *, eps=1e-6):
        rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)

        return (input / rms) * weight

    rms_norm_fn = getattr(torch.nn.functional, "rms_norm", _fallback)

    result = rms_norm_fn(input, input.shape[-1:], weight=weight, eps=eps)

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
