import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "top_k_value, top_p_value, allowed",
    (
        (3, 1.0, (0, 1, 2)),
        (16, 0.6, (0,)),
        (3, 0.8, (0, 1)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampling_from_logits(
    top_k_value,
    top_p_value,
    allowed,
    dtype,
    device,
    implementation_index,
):
    batch_size = 64
    logits = torch.full((batch_size, 16), -10.0, dtype=dtype, device=device)
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    top_k = torch.full((batch_size,), top_k_value, dtype=torch.int64)
    top_p = torch.full((batch_size,), top_p_value, dtype=torch.float32)
    first = torch.empty((batch_size,), dtype=torch.int32, device=device)
    second = torch.empty_like(first)
    different_seed = torch.empty_like(first)

    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, first, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, second, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 5678, 9, different_seed, implementation_index
    )

    assert torch.equal(first, second)
    if len(allowed) > 1:
        assert not torch.equal(first, different_seed)
    allowed_tensor = torch.tensor(allowed, dtype=torch.int32, device=device)
    assert torch.all(torch.isin(first, allowed_tensor))


def test_flashinfer_sampling_joint_host_indices(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        (
            (9.0, 1.0, 0.0, -1.0),
            (0.0, 8.0, 1.0, -1.0),
            (-1.0, 0.0, 1.0, 7.0),
        ),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor((2, 0, 2, 1, 0, 1, 2), dtype=torch.int64)
    batch_size = indices.numel()
    top_k = torch.ones(batch_size, dtype=torch.int64)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        out,
        implementation_index,
        indices=indices,
        filter_apply_order="joint",
    )

    expected = torch.tensor((3, 0, 3, 1, 0, 1, 3), dtype=torch.int64, device=device)
    assert torch.equal(out, expected)


def test_flashinfer_sampling_top_k_first_cuda_indices(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        (
            (9.0, 1.0, 0.0, -1.0),
            (0.0, 8.0, 1.0, -1.0),
            (-1.0, 0.0, 1.0, 7.0),
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.tensor((2, 0, 2, 1, 0), dtype=torch.int32, device=device)
    batch_size = indices.numel()
    top_k = torch.ones(batch_size, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float64)
    out = torch.empty(batch_size, dtype=torch.int32, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        out,
        implementation_index,
        indices=indices,
        filter_apply_order="top_k_first",
    )

    expected = torch.tensor((3, 0, 3, 1, 0), dtype=torch.int32, device=device)
    assert torch.equal(out, expected)


def test_flashinfer_sampling_offset(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    batch_size = 256
    logits = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    top_k = torch.full((batch_size,), 4, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    first = torch.empty(batch_size, dtype=torch.int32, device=device)
    repeated = torch.empty_like(first)
    different_offset = torch.empty_like(first)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        first,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        repeated,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        10,
        different_offset,
        implementation_index,
        filter_apply_order="joint",
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_offset)


def test_flashinfer_sampling_uses_handle_stream(device, implementation_index):
    if device != "cuda" or implementation_index != 16:
        pytest.skip("FlashInfer linked-provider stream coverage")

    batch_size = 64
    logits = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    logits[:, 0] = 1.0
    top_k = torch.ones(batch_size, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    out = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream()

    def call_sampling():
        infini.ops.top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            None,
            "joint",
            True,
            False,
            1234,
            9,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

    try:
        call_sampling()
        stream.synchronize()
        out.fill_(-1)
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
        call_sampling()

        default_stream = torch.cuda.default_stream()
        with torch.cuda.stream(default_stream):
            snapshot = out.clone()
        default_stream.synchronize()
        assert torch.all(snapshot == -1)

        stream.synchronize()
        assert torch.all(out == 0)
    finally:
        torch.cuda.synchronize()


def _top_k_top_p_sampling_from_logits(
    logits,
    top_k,
    top_p,
    seed,
    offset,
    out,
    implementation_index,
    *,
    indices=None,
    filter_apply_order="top_k_first",
):
    infini.ops.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        indices,
        filter_apply_order,
        True,
        False,
        seed,
        offset,
        out,
        stream=get_stream(logits.device),
        implementation_index=implementation_index,
    )
