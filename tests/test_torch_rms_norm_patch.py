import infini.ops
import infini.torch
import pytest
import torch


def _rms_norm_reference(input, weight, eps):
    square_mean = torch.mean(
        input.float() * input.float(), dim=-1, keepdim=True
    )
    rstd = torch.rsqrt(square_mean + eps)

    return (input.float() * rstd * weight.float()).to(input.dtype)


def _skip_unstable_metax_dtype(dtype):
    if "metax" in torch.__version__.lower() and dtype != torch.float32:
        pytest.skip("metax rms_norm replacement is currently limited to float32")


def _skip_missing_functional_rms_norm():
    if not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("torch.nn.functional.rms_norm is not available")


def _spy_rms_norm(monkeypatch):
    calls = []
    original = infini.ops.rms_norm

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))

        return original(*args, **kwargs)

    monkeypatch.setattr(infini.ops, "rms_norm", wrapper)

    return calls


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_rms_norm_patch_calls_infini_ops(monkeypatch, dtype, rtol, atol):
    _skip_missing_functional_rms_norm()
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")

    _skip_unstable_metax_dtype(dtype)

    input = torch.randn((2, 4, 64), device="cuda", dtype=dtype)
    weight = torch.randn((64,), device="cuda", dtype=dtype)
    eps = 1e-6
    calls = _spy_rms_norm(monkeypatch)

    with infini.torch.patch():
        actual = torch.nn.functional.rms_norm(
            input, input.shape[-1:], weight=weight, eps=eps
        )

    expected = _rms_norm_reference(input, weight, eps)

    assert calls
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def test_rms_norm_patch_is_scoped(monkeypatch):
    _skip_missing_functional_rms_norm()
    input = torch.randn((2, 4, 64))
    weight = torch.randn((64,))
    calls = _spy_rms_norm(monkeypatch)

    torch.nn.functional.rms_norm(input, input.shape[-1:], weight=weight, eps=1e-6)

    assert not calls


def test_rms_norm_patch_fallback_for_unsupported_shape(monkeypatch):
    _skip_missing_functional_rms_norm()
    input = torch.randn((2, 4, 8, 8))
    weight = torch.randn((8, 8))
    calls = _spy_rms_norm(monkeypatch)

    with infini.torch.patch():
        actual = torch.nn.functional.rms_norm(input, (8, 8), weight=weight, eps=1e-6)

    expected = torch.nn.functional.rms_norm(input, (8, 8), weight=weight, eps=1e-6)

    assert not calls
    assert torch.allclose(actual, expected)
