import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


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
        (grad, learning_rate, beta1, beta2, eps, step, param, m, v),
        {},
        rtol=rtol,
        atol=atol,
    )


def _adam_accumulate_grad(
    grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    param,
    m,
    v,
    *,
    implementation_index,
):
    infini.ops.adam_accumulate_grad(
        grad,
        learning_rate,
        beta1,
        beta2,
        eps,
        step,
        param,
        m,
        v,
        stream=get_stream(grad.device),
        implementation_index=implementation_index,
    )

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))


def _torch_adam_accumulate_grad(
    grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    param,
    m,
    v,
):
    reference_param = torch.nn.Parameter(
        param.detach().to(device="cpu", dtype=torch.float64)
    )
    optimizer = torch.optim.Adam(
        [reference_param],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        foreach=False,
    )
    optimizer.state[reference_param] = {
        "step": torch.tensor(float(step - 1)),
        "exp_avg": m.detach().to(device="cpu", dtype=torch.float64).clone(),
        "exp_avg_sq": v.detach().to(device="cpu", dtype=torch.float64).clone(),
    }
    reference_param.grad = grad.detach().to(device="cpu", dtype=torch.float64).clone()
    optimizer.step()
    param.copy_(reference_param.detach())
    m.copy_(optimizer.state[reference_param]["exp_avg"])
    v.copy_(optimizer.state[reference_param]["exp_avg_sq"])

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))
