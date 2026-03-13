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
@pytest.mark.parametrize("alpha", (-1, -0.5, 0, 0.5, 1))
@pytest.mark.parametrize("beta", (-1, -0.5, 0, 0.5, 1))
@pytest.mark.parametrize("trans_a", (False, True))
@pytest.mark.parametrize("trans_b", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
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
    # Skip transposing test cases for MLU platform as transposing is not currently supported.
    if device == "mlu" and (trans_a or trans_b):
        pytest.skip("transposing is not currently supported on MLU")

    a = randn_strided(a_shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(b_shape, b_strides, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    c = randn_strided(c_shape, c_strides, dtype=dtype, device=device)

    return Payload(
        _gemm,
        _torch_gemm,
        (a, b, alpha, beta, trans_a, trans_b, c),
        {},
        rtol=rtol,
        atol=atol,
    )


def _gemm(a, b, alpha, beta, trans_a, trans_b, c):
    infini.ops.gemm(a, b, alpha, beta, trans_a, trans_b, c)

    return c


def _torch_gemm(a, b, alpha=1.0, beta=1.0, trans_a=False, trans_b=False, c=None):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    # PyTorch `baddbmm`/`addmm` ignores `beta` when `alpha=0.0`.
    if alpha == 0:
        c.mul_(beta)

        return c

    # Some backends (e.g. `torch_musa`) may reject `addmm`/`baddbmm(out=...)`
    # for certain strided outputs. Fall back to `matmul` plus fused `alpha`/`beta`
    # update to keep reference coverage.
    try:
        if a.ndim == 2:
            return torch.addmm(c, a, b, beta=beta, alpha=alpha, out=c)

        return torch.baddbmm(c, a, b, beta=beta, alpha=alpha, out=c)
    except RuntimeError:
        c_original = c.clone()
        torch.matmul(a, b, out=c)
        c.mul_(alpha).add_(c_original, alpha=beta)

        return c
