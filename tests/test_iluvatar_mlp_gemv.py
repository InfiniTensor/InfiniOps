import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("m", (1, 2))
@pytest.mark.parametrize("k,n", ((4096, 32768), (16384, 4096)))
def test_iluvatar_mlp_gemv(m, k, n, device):
    if device != "cuda" or not infini.ops.Gemm.active_implementation_indices(
        "iluvatar"
    ):
        pytest.skip("requires the Iluvatar backend")

    # Bounded dyadic values exercise both signs without tolerance masking errors.
    x = ((torch.arange(m * k, device=device) % 7 - 3).float() / 16).reshape(m, k)
    w = ((torch.arange(n * k, device=device) % 5 - 2).float() / 16).reshape(n, k)
    a = x.to(torch.bfloat16)
    b = w.to(torch.bfloat16).t()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    infini.ops.gemm(
        a,
        b,
        None,
        1.0,
        0.0,
        False,
        False,
        out,
        stream=get_stream(a.device),
        implementation_index=0,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # Bias/scalar calls must retain BLAS semantics outside the specialization.
    bias = torch.ones_like(out)
    infini.ops.gemm(
        a,
        b,
        bias,
        0.5,
        1.0,
        False,
        False,
        out,
        stream=get_stream(a.device),
        implementation_index=0,
    )
    expected = (0.5 * (a.float() @ b.float()) + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
