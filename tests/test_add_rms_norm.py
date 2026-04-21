import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape",
    (
        (1, 64),
        (2, 128),
        (4, 48, 64),
        (2, 4, 2048),
    ),
)
# TODO: Generate implementation indices dynamically.
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_add_rms_norm(shape, implementation_index, eps, dtype, device, rtol, atol):
    active_indices = infini.ops.AddRmsNorm.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    input = randn_strided(shape, None, dtype=dtype, device=device)
    other = randn_strided(shape, None, dtype=dtype, device=device)
    weight = randn_strided((shape[-1],), None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)
    residual_out = empty_strided(shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _add_rms_norm(
            *args, implementation_index=implementation_index, **kwargs
        ),
        _torch_add_rms_norm,
        (input, other, weight),
        {"eps": eps, "out": out, "residual_out": residual_out},
        rtol=rtol,
        atol=atol,
    )


def _add_rms_norm(input, other, weight, *, eps, out, residual_out, implementation_index):
    infini.ops.add_rms_norm(
        input, other, weight, eps, out, residual_out,
        implementation_index=implementation_index,
    )

    # Concatenate both outputs so `auto_act_and_assert` checks both.
    return torch.cat((out.reshape(-1), residual_out.reshape(-1)))


def _torch_add_rms_norm(input, other, weight, *, eps, out, residual_out):
    def _fallback(x, _shape, weight, *, eps):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)

        return (x / rms) * weight

    rms_norm_fn = getattr(torch.nn.functional, "rms_norm", _fallback)
    residual_out.copy_(input + other)
    out.copy_(rms_norm_fn(residual_out, input.shape[-1:], weight=weight, eps=eps))

    return torch.cat((out.reshape(-1), residual_out.reshape(-1)))
