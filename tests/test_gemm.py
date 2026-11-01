import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, y_shape, a_strides, b_strides, y_strides",
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
    y_shape,
    a_strides,
    b_strides,
    y_strides,
    alpha,
    beta,
    trans_a,
    trans_b,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    # Skip transposing test cases for MLU platform as transposing is not currently supported.
    if device == "mlu" and (trans_a or trans_b):
        pytest.skip("transposing is not currently supported on MLU")

    # `cnnlBatchMatMulEx` does not accept `bfloat16` inputs on MLU.
    if device == "mlu" and dtype == torch.bfloat16:
        pytest.skip("`bfloat16` is not supported by `cnnlBatchMatMulEx`")

    if implementation_index == 1 and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("cuBLASLt half-precision exceeds current tolerances")

    if (
        implementation_index == 2
        and device == "cpu"
        and dtype in (torch.float16, torch.bfloat16)
    ):
        pytest.skip("ATen CPU `addmm`/`baddbmm` does not support half-precision")

    if (
        device == "cuda"
        and dtype == torch.float16
        and infini.ops.Gemm.active_implementation_indices("iluvatar")
    ):
        pytest.skip("Iluvatar GEMM reports fp16 execution failures")

    a = randn_strided(a_shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(b_shape, b_strides, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    y = randn_strided(y_shape, y_strides, dtype=dtype, device=device)
    use_portable_ref = implementation_index == 2 and not (
        device == "cpu"
        or (
            device == "cuda" and infini.ops.Gemm.active_implementation_indices("nvidia")
        )
    )
    ref = _torch_gemm_portable if use_portable_ref else _torch_gemm

    return Payload(
        lambda *args: _gemm(*args, implementation_index=implementation_index),
        ref,
        (a, b, None, alpha, beta, trans_a, trans_b, y),
        {},
        rtol=rtol,
        atol=atol,
    )


def _gemm(a, b, c, alpha, beta, trans_a, trans_b, y, implementation_index=0):
    infini.ops.gemm(
        a,
        b,
        c,
        alpha,
        beta,
        trans_a,
        trans_b,
        y,
        stream=get_stream(a.device),
        implementation_index=implementation_index,
    )

    return y


def _torch_gemm(a, b, c, alpha, beta, trans_a, trans_b, y):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    if c is None:
        beta = 0.0
        y.zero_()
        c = y

    # PyTorch `baddbmm`/`addmm` ignores `beta` when `alpha=0.0`.
    if alpha == 0:
        y.copy_(c)
        y.mul_(beta)

        return y

    # Some backends (e.g. `torch_musa`) may reject `addmm`/`baddbmm(out=...)`
    # for certain strided outputs. Fall back to `matmul` plus fused `alpha`/`beta`
    # update to keep reference coverage.
    try:
        if a.ndim == 2:
            return torch.addmm(c, a, b, beta=beta, alpha=alpha, out=y)

        return torch.baddbmm(c, a, b, beta=beta, alpha=alpha, out=y)
    except RuntimeError:
        # Fallback for backends that don't support `addmm`/`baddbmm` (e.g. CPU `float16`/`bfloat16`):
        # compute in float32 and cast back.
        c_original = c.float()
        result = torch.matmul(a.float(), b.float())
        y.copy_((alpha * result + beta * c_original).to(y.dtype))

        return y


def _torch_gemm_portable(a, b, c, alpha, beta, trans_a, trans_b, y):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    if c is None:
        beta = 0.0
        y.zero_()
        c = y

    if alpha == 0:
        y.copy_(c)
        y.mul_(beta)

        return y

    product = torch.matmul(a, b)
    if beta == 0:
        y.copy_(product)
        y.mul_(alpha)

        return y

    y.copy_(c)
    y.mul_(beta)
    y.add_(product, alpha=alpha)

    return y


@pytest.mark.smoke
def test_aten_external_stream(device, implementation_index):
    if implementation_index != 2 or device == "cpu":
        pytest.skip("Requires the accelerator ATen provider")
    accelerator = getattr(torch, torch.device(device).type)
    previous = get_stream(device)
    stream = accelerator.Stream()
    raw = getattr(stream, f"{torch.device(device).type}_stream")
    x = torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3)
    y = torch.arange(6, device=device, dtype=torch.float32).reshape(3, 2)
    out = torch.full((2, 2), -123.0, device=device)
    reference = x @ y
    initial = out.clone()
    stream.wait_stream(accelerator.current_stream())

    def invoke():
        infini.ops.gemm(
            x, y, out, implementation_index=implementation_index, stream=raw
        )

    try:
        invoke()
        accelerator.synchronize()
        out.copy_(initial)
        accelerator.synchronize()
        delayed = None
        if device == "cuda":
            accelerator._sleep(200_000_000)
            delayed = accelerator.Event()
            delayed.record()
        invoke()
        with accelerator.stream(stream):
            snapshot = out.clone()
        stream.synchronize()
        if delayed is not None:
            assert not delayed.query(), (
                "The unrelated stream completed before the snapshot"
            )
        assert get_stream(device) == previous
    finally:
        accelerator.synchronize()
    torch.testing.assert_close(snapshot, reference, rtol=1e-4, atol=1e-5)
