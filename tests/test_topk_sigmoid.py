import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64, torch.uint32))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-6, 1e-6),
        (torch.bfloat16, 1e-6, 1e-6),
    ),
)
def test_topk_sigmoid(dtype, index_dtype, has_bias, renormalize, rtol, atol):
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (1.25, -0.5, 0.75, 2.0, -1.0),
            (-0.25, 1.5, 0.5, -1.25, 2.25),
            (0.125, 0.75, 2.5, 1.0, -0.75),
        ),
        dtype=dtype,
        device="cuda",
    )
    e_score_correction_bias = None
    if has_bias:
        e_score_correction_bias = torch.tensor(
            (0.0, 0.75, -0.5, -1.0, 1.25),
            dtype=torch.float32,
            device=gating_output.device,
        )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=index_dtype)

    result = infini.ops.topk_sigmoid(
        gating_output,
        e_score_correction_bias,
        None,
        renormalize,
        0.75,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    expected = _reference(
        gating_output, e_score_correction_bias, None, 2, renormalize, 0.75
    )
    torch.testing.assert_close(outputs[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_bias_only_changes_selection():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(((3.0, 2.0, 1.0),), dtype=torch.float32, device="cuda")
    e_score_correction_bias = torch.tensor(
        (-4.0, 0.0, 3.0), dtype=torch.float32, device="cuda"
    )
    outputs = _make_outputs(gating_output, topk=1, index_dtype=torch.int32)

    infini.ops.topk_sigmoid(
        gating_output,
        e_score_correction_bias,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    unbiased_scores = torch.sigmoid(gating_output.to(torch.float32))
    assert outputs[1].item() == 2
    torch.testing.assert_close(outputs[0], unbiased_scores[:, 2:3])


def test_topk_sigmoid_nan_tie_uses_lower_expert_ids():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.full((1, 4), torch.nan, dtype=torch.float32, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int32)

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    torch.testing.assert_close(outputs[0], torch.zeros_like(outputs[0]))
    torch.testing.assert_close(
        outputs[1],
        torch.tensor(((0, 1),), dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64, torch.uint32))
def test_topk_sigmoid_padding_and_token_expert_indices(index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (0.25, 2.0, -0.5, 1.0),
            (1.5, -0.25, 0.75, 0.0),
            (-1.0, 0.5, 2.25, 1.25),
        ),
        dtype=torch.float16,
        device="cuda",
    )
    is_padding = torch.tensor((False, True, False), dtype=torch.bool, device="cuda")
    outputs = _make_outputs(gating_output, topk=3, index_dtype=index_dtype)

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        is_padding,
        True,
        1.5,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    expected = _reference(gating_output, None, is_padding, 3, True, 1.5)
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_non_default_stream():
    if not torch.cuda.is_available():
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    gating_output = torch.randn((7, 13), dtype=torch.bfloat16, device="cuda")
    outputs = _make_outputs(gating_output, topk=4, index_dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        0.5,
        *outputs,
        stream=stream.cuda_stream,
    )

    stream.synchronize()
    expected = _reference(gating_output, None, None, 4, False, 0.5)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs[1], expected[1].to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_empty_tokens():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.empty((0, 4), dtype=torch.float16, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int64)

    result = infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    assert all(output.shape == (0, 2) for output in outputs)


def _make_outputs(gating_output, topk, index_dtype):
    shape = (gating_output.size(0), topk)
    topk_weights = torch.full(
        shape, torch.nan, dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.full(
        shape, 2**31 - 1, dtype=index_dtype, device=gating_output.device
    )
    token_expert_indices = torch.full(
        shape, -1, dtype=torch.int32, device=gating_output.device
    )

    return topk_weights, topk_ids, token_expert_indices


def _reference(
    gating_output,
    e_score_correction_bias,
    is_padding,
    topk,
    renormalize,
    routed_scaling_factor,
):
    scores = torch.sigmoid(gating_output.to(torch.float32))
    selection_scores = (
        scores if e_score_correction_bias is None else scores + e_score_correction_bias
    )
    ids = torch.topk(selection_scores, topk, dim=-1).indices
    weights = scores.gather(1, ids)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(denominator > 0, denominator, 1.0)
    weights = weights * routed_scaling_factor

    if is_padding is not None:
        ids = ids.masked_fill(is_padding.bool().unsqueeze(-1), -1)

    num_tokens = gating_output.size(0)
    token_expert_indices = torch.arange(
        topk, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(0).expand(num_tokens, -1) * num_tokens + torch.arange(
        num_tokens, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(-1)

    return weights, ids, token_expert_indices
