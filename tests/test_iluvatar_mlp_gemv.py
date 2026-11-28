import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.smoke
@pytest.mark.parametrize("m", (1, 2))
@pytest.mark.parametrize(
    "k,n",
    (
        (2048, 4096),
        (4160, 2052),
        (4224, 2056),
        (4288, 2060),
        (2048, 11008),
        (5504, 2048),
        (3072, 16384),
        (8192, 3072),
        (4096, 22016),
        (11008, 4096),
        (4096, 28672),
        (14336, 4096),
        (5120, 27648),
        (13824, 5120),
        (8192, 57344),
        (28672, 8192),
        (4096, 32768),
        (16384, 4096),
    ),
)
@pytest.mark.parametrize("implementation_index", (0,))
def test_iluvatar_mlp_gemv(m, k, n, device, implementation_index):
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
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # Non-default alpha must retain the supported BLAS fallback.
    # The current base GEMM contract requires C=None and forces beta=0.
    infini.ops.gemm(
        a,
        b,
        None,
        0.5,
        0.0,
        False,
        False,
        out,
        stream=get_stream(a.device),
        implementation_index=implementation_index,
    )
    expected = (0.5 * (a.float() @ b.float())).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    a = (torch.randn((m, k), device=device) / k**0.5).to(torch.bfloat16)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16).t()
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
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m,k,n,layout",
    (
        (1, 1024, 1024, "column"),
        (2, 1088, 1028, "column"),
        (3, 2048, 4096, "column"),
        (1, 768, 16384, "column"),
        (1, 2049, 4096, "column"),
        (1, 2048, 4097, "column"),
        (1, 2048, 4096, "row"),
        (2, 2048, 4096, "padded_a"),
        (1, 2048, 4096, "padded_b"),
    ),
)
@pytest.mark.parametrize("implementation_index", (0,))
def test_iluvatar_mlp_gemv_fallback(m, k, n, layout, device, implementation_index):
    if device != "cuda" or not infini.ops.Gemm.active_implementation_indices(
        "iluvatar"
    ):
        pytest.skip("requires the Iluvatar backend")

    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16).t()

    if layout == "row":
        b = b.contiguous()
    elif layout == "padded_a":
        storage = torch.empty((m, k + 8), device=device, dtype=torch.bfloat16)
        a = storage[:, :k].copy_(a)
    elif layout == "padded_b":
        storage = torch.empty((n, k + 8), device=device, dtype=torch.bfloat16)
        b = storage[:, :k].t().copy_(b)

    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
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
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
