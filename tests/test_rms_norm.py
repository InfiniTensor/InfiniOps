import infini.ops
import pytest

import torch
from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, weight_strides, out_strides",
    (
        ((1, 64), (64,), None, None, None),
        ((2, 128), (128,), None, None, None),
        ((4, 48, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
        ((1, 64), (64,), (64, 1), (1,), (64, 1)),
        ((4, 48, 64), (64,), (3072, 64, 1), (1,), (3072, 64, 1)),
    ),
)
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_rms_norm(
    input_shape,
    weight_shape,
    input_strides,
    weight_strides,
    out_strides,
    eps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "npu" and implementation_index == 9 and dtype == torch.bfloat16:
        pytest.skip("Ascend NineToothed RMSNorm does not support bfloat16 yet")

    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(input_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _rms_norm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_rms_norm,
        (input, weight),
        {"eps": eps, "out": out},
        rtol=rtol,
        atol=atol,
    )


def test_rms_norm_non_default_stream(device, implementation_index):
    if device == "cuda":
        accelerator = torch.cuda
        stream_attribute = "cuda_stream"
    elif device == "musa":
        accelerator = torch.musa
        stream_attribute = "musa_stream"
    elif device == "mlu":
        accelerator = torch.mlu
        stream_attribute = "mlu_stream"
    else:
        pytest.skip("non-default streams require an accelerator backend")

    input = torch.randn((32, 128), dtype=torch.float16, device=device)
    weight = torch.randn((128,), dtype=torch.float16, device=device)
    out = torch.zeros_like(input)
    expected = _torch_rms_norm(input, weight, out=torch.empty_like(out)).cpu()
    accelerator.synchronize()

    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())
    stream_ptr = getattr(stream, stream_attribute)

    accelerator._sleep(50_000_000)
    try:
        infini.ops.rms_norm(
            input,
            weight,
            1e-6,
            out,
            implementation_index=implementation_index,
            stream=stream_ptr,
        )
        stream.synchronize()
        with accelerator.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        accelerator.synchronize()


@pytest.mark.parametrize("seq_len", (1, 36, 333))
@pytest.mark.parametrize("heads, offset", ((16, 0), (8, 2048)))
@pytest.mark.parametrize("strided_out", (False, True))
@pytest.mark.parametrize("transpose_rows", (False, True))
def test_rms_norm_qkv_slice(
    seq_len, heads, offset, strided_out, transpose_rows, device, implementation_index
):
    if device != "npu":
        pytest.skip("regression for Ascend RMSNorm staging")

    width = heads * 128
    weight_cpu = torch.randn(128, dtype=torch.float16)
    weight = weight_cpu.to(device)

    # Reuse cached operators with different allocations and values.
    for _ in range(3):
        source_cpu = torch.randn(seq_len, 4096, dtype=torch.float16)
        source = source_cpu.to(device)
        input = source[:, offset : offset + width].view(seq_len, heads, 128)
        expected = _torch_rms_norm(
            source_cpu[:, offset : offset + width].reshape(seq_len, heads, 128),
            weight_cpu,
        )

        if strided_out:
            backing = torch.full(
                (seq_len, width + 256), -123.0, dtype=torch.float16, device=device
            )
            out = backing[:, 128 : 128 + width].view(seq_len, heads, 128)
        else:
            out = torch.empty((seq_len, heads, 128), dtype=torch.float16, device=device)

        if transpose_rows:
            input = input.transpose(0, 1)
            out = out.transpose(0, 1)
            expected = expected.transpose(0, 1)

        _rms_norm(input, weight, out=out, implementation_index=implementation_index)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(source.cpu(), source_cpu, rtol=0, atol=0)

        if strided_out:
            actual_backing = backing.cpu()
            expected_backing = torch.full(
                (seq_len, width + 256), -123.0, dtype=torch.float16
            )

            if transpose_rows:
                expected = expected.transpose(0, 1)

            expected_backing[:, 128 : 128 + width] = expected.reshape(seq_len, width)
            torch.testing.assert_close(
                actual_backing, expected_backing, rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                actual_backing[:, :128], expected_backing[:, :128], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_backing[:, -128:], expected_backing[:, -128:], rtol=0, atol=0
            )


def test_rms_norm_strided_npu_stream(device, implementation_index):
    if device != "npu":
        pytest.skip("regression for Ascend RMSNorm staging streams")

    source_cpu = torch.randn(36, 4096, dtype=torch.float16)
    weight_cpu = torch.randn(128, dtype=torch.float16)
    expected = _torch_rms_norm(source_cpu[:, 2048:3072].reshape(36, 8, 128), weight_cpu)
    stream = torch.npu.Stream()

    with torch.npu.stream(stream):
        source = source_cpu.to(device)
        weight = weight_cpu.to(device)
        backing = torch.full((36, 1280), -123.0, dtype=torch.float16, device=device)
        input = source[:, 2048:3072].view(36, 8, 128)
        out = backing[:, 128:1152].view(36, 8, 128)
        _rms_norm(input, weight, out=out, implementation_index=implementation_index)

    stream.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-2)


def _rms_norm(input, weight, *, eps=1e-6, out=None, implementation_index=0):
    infini.ops.rms_norm(
        input,
        weight,
        eps,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_rms_norm(input, weight, *, eps=1e-6, out=None):
    # Fallback for `torch<2.3`: `rms_norm = (x / sqrt(mean(x^2) + eps)) * weight`.
    def _fallback(input, _normalized_shape, weight, *, eps=1e-6):
        rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)

        return (input / rms) * weight

    rms_norm_fn = getattr(torch.nn.functional, "rms_norm", _fallback)

    result = rms_norm_fn(input, input.shape[-1:], weight=weight, eps=eps)

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
