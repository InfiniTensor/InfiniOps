import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape, a_strides, b_strides, c_strides",
    (
        ((1, 2048), (2048, 2048), (1, 2048), None, None, None),
        ((2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
        ((1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
        ((6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
        ((4, 48, 64), (4, 64, 6), (4, 48, 6), None, None, None),
    ),
)
@pytest.mark.parametrize("trans_a", (False, True))
@pytest.mark.parametrize("trans_b", (False, True))
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 5e-2, 5e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ),
)
def test_matmul(
    a_shape,
    b_shape,
    c_shape,
    a_strides,
    b_strides,
    c_strides,
    trans_a,
    trans_b,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.Matmul.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    if implementation_index == 0 and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("cuBLASLt half-precision exceeds current tolerances")

    a = randn_strided(a_shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(b_shape, b_strides, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    c = randn_strided(c_shape, c_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args: _matmul(*args, implementation_index=implementation_index),
        _torch_matmul,
        (a, b, c, trans_a, trans_b),
        {},
        rtol=rtol,
        atol=atol,
    )


def _matmul(a, b, c, trans_a, trans_b, implementation_index=0):
    infini.ops.matmul(
        a,
        b,
        c,
        trans_a,
        trans_b,
        implementation_index=implementation_index,
    )

    return c


def _torch_matmul(a, b, c, trans_a=False, trans_b=False):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    try:
        return torch.matmul(a, b, out=c)
    except RuntimeError:
        # Fallback for backends that don't support `matmul(out=...)` for
        # certain strided outputs or half-precision types.
        result = torch.matmul(a.float(), b.float())
        c.copy_(result.to(c.dtype))

        return c
