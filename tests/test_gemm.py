import ops
import pytest
import torch

from tests.utils import empty_strided, get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
# TODO: Add support for more data types.
@pytest.mark.parametrize("dtype, rtol, atol", ((torch.float32, 1e-3, 1e-3),))
@pytest.mark.parametrize("trans_b", (False, True))
@pytest.mark.parametrize("trans_a", (False, True))
@pytest.mark.parametrize("beta", (-1, -0.5, 0, 0.5, 1))
@pytest.mark.parametrize("alpha", (-1, -0.5, 0, 0.5, 1))
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
def test_gemm(
    a_shape,
    b_shape,
    c_shape,
    a_strides,
    b_strides,
    c_strides,
    alpha,
    beta,
    trans_a,
    trans_b,
    dtype,
    device,
    rtol,
    atol,
):
    a = empty_strided(a_shape, a_strides, dtype=dtype, device=device)
    b = empty_strided(b_shape, b_strides, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    output = empty_strided(c_shape, c_strides, dtype=dtype, device=device)
    expected = output.clone()

    a.normal_()
    b.normal_()

    # TODO: Add keyword argument support.
    ops.gemm(a, b, alpha, beta, trans_a, trans_b, output)
    _torch_gemm(
        a, b, alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b, c=expected
    )

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


def _torch_gemm(a, b, *, alpha=1.0, beta=1.0, trans_a=False, trans_b=False, c=None):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    if a.ndim == 2:
        return torch.addmm(c, a, b, beta=beta, alpha=alpha, out=c)

    return torch.baddbmm(c, a, b, beta=beta, alpha=alpha, out=c)
