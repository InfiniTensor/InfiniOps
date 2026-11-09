import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((3, 3), None, None),
        ((3, 5), None, None),
        ((32, 512), None, None),
        ((32, 512), (1024, 1), (1024, 1)),
        ((4, 20, 512), None, None),
        ((4, 20, 512), (20480, 512, 1), None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_causal_softmax(shape, input_strides, out_strides, dtype, device, rtol, atol):
    input_tensor = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _causal_softmax,
        _torch_causal_softmax,
        (input_tensor, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _causal_softmax(input, out):
    infini.ops.causal_softmax(input, out, stream=get_stream(input.device))

    return out


def _torch_causal_softmax(input, out):
    input_cpu = input.detach().cpu().to(torch.float32)
    mask = torch.tril(torch.ones_like(input_cpu), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, -torch.inf, input_cpu)
    result = torch.nn.functional.softmax(masked, dim=-1)
    out.copy_(result.to(device=out.device, dtype=out.dtype))

    return out


@pytest.mark.smoke
@pytest.mark.parametrize("q,k", ((2, 3), (521, 521), (2051, 2051), (521, 4096)))
@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize("implementation_index", (0,))
def test_iluvatar_causal_fp32(q, k, inplace, device, implementation_index):
    if (
        device != "cuda"
        or not infini.ops.CausalSoftmaxInfinilm.active_implementation_indices(
            "iluvatar"
        )
    ):
        pytest.skip("requires the Iluvatar backend")

    # Include masked NaNs: the causal branch must select them away.
    index = torch.arange(2 * q * k, device=device).reshape(2, q, k)
    source = ((index % 29 - 14).float() / 8).to(torch.bfloat16)
    mask = torch.arange(k, device=device)[None, :] >= (
        k - q + torch.arange(q, device=device)[:, None] + 1
    )
    source.masked_fill_(mask, float("nan"))
    reference = source.clone().masked_fill_(mask, float("-inf"))
    # Match the FP32 sum before BF16 exponent storage of the InfiniLM path.
    infini.ops.softmax(reference, -1, None, reference, stream=get_stream(source.device))
    out = source if inplace else torch.empty_like(source)
    infini.ops.causal_softmax_infinilm(
        source,
        out,
        stream=get_stream(source.device),
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out.masked_select(mask)) == 0
