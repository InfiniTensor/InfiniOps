import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (shape, learning_rate, beta1, beta2, eps, step, strides)
_TEST_CASES = (
    pytest.param(
        (17,),
        1e-2,
        0.9,
        0.999,
        1e-8,
        1,
        None,
        id="one-dimensional-first-step",
    ),
    pytest.param(
        (2, 3),
        5e-3,
        0.8,
        0.99,
        1e-6,
        3,
        None,
        id="two-dimensional-third-step",
    ),
    pytest.param(
        (2, 3, 4),
        1e-2,
        0.9,
        0.999,
        1e-8,
        2,
        (30, 7, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, learning_rate, beta1, beta2, eps, step, strides",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_adam_accumulate_grad(
    shape,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad = randn_strided(shape, strides, dtype=dtype, device=device)
    param = randn_strided(shape, strides, dtype=dtype, device=device)
    m = randn_strided(shape, strides, dtype=dtype, device=device)
    v = randn_strided(shape, strides, dtype=dtype, device=device).abs_().add_(0.1)

    return Payload(
        lambda *args: _adam_accumulate_grad(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_adam_accumulate_grad,
        (grad, param, m, v, learning_rate, beta1, beta2, eps, step),
        {},
        rtol=rtol,
        atol=atol,
    )


def _adam_accumulate_grad(
    grad,
    param,
    m,
    v,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    *,
    implementation_index,
):
    infini.ops.adam_accumulate_grad(
        grad,
        param,
        m,
        v,
        learning_rate,
        beta1,
        beta2,
        eps,
        step,
        stream=get_stream(grad.device),
        implementation_index=implementation_index,
    )

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))


def _torch_adam_accumulate_grad(
    grad,
    param,
    m,
    v,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
):
    m.copy_(beta1 * m + (1 - beta1) * grad)
    v.copy_(beta2 * v + (1 - beta2) * grad * grad)

    bias_correction_m = 1 - beta1**step
    bias_correction_v = 1 - beta2**step
    m_hat = m.float() / bias_correction_m
    v_hat = v.float() / bias_correction_v
    update = learning_rate * m_hat / (torch.sqrt(v_hat) + eps)
    param.copy_(param - update.to(dtype=param.dtype))

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))
