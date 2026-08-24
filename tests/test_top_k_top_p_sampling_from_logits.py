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


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampling_from_logits_per_request_filters(
    dtype,
    device,
    implementation_index,
):
    logits = torch.tensor(
        (
            (5.0, 4.0, 3.0, -10.0),
            (5.0, 4.0, 3.0, -10.0),
            (5.0, 4.0, 3.0, -10.0),
            (2.0, 1.9, 1.8, 1.7),
        ),
        dtype=dtype,
        device=device,
    )
    top_k = torch.tensor((1, 3, 2, 2), dtype=torch.int32)
    top_p = torch.tensor((1.0, 0.8, 1.0, 0.4), dtype=torch.float64)
    out = torch.empty((4,), dtype=torch.int32, device=device)

    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, out, implementation_index
    )

    assert out[0].item() == 0
    assert out[1].item() in (0, 1)
    assert out[2].item() in (0, 1)
    assert out[3].item() == 0


def test_top_k_top_p_sampling_from_logits_top_k_bounds_and_singleton(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA edge-case coverage requires CUDA")

    batch_size = 64
    vocab_size = 32
    logits = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    top_p = torch.ones((batch_size,), dtype=torch.float64)
    outputs = []

    for top_k_value in (0, -3, vocab_size + 7):
        top_k = torch.full((batch_size,), top_k_value, dtype=torch.int64)
        out = torch.empty((batch_size,), dtype=torch.int32, device=device)
        _top_k_top_p_sampling_from_logits(
            logits, top_k, top_p, 1234, 9, out, implementation_index
        )
        outputs.append(out)

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])

    singleton_logits = torch.randn((batch_size, 1), dtype=torch.float32, device=device)
    singleton_top_k = torch.zeros((batch_size,), dtype=torch.int32)
    singleton_out = torch.empty((batch_size,), dtype=torch.int32, device=device)
    _top_k_top_p_sampling_from_logits(
        singleton_logits,
        singleton_top_k,
        top_p,
        1234,
        9,
        singleton_out,
        implementation_index,
    )

    assert torch.count_nonzero(singleton_out).item() == 0


def test_top_k_top_p_sampling_from_logits_large_offset(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA counter-based RNG coverage requires CUDA")

    batch_size = 64
    vocab_size = 16
    logits = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    top_k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
    top_p = torch.ones((batch_size,), dtype=torch.float64)
    first = torch.empty((batch_size,), dtype=torch.int32, device=device)
    shifted = torch.empty_like(first)
    repeated = torch.empty_like(first)
    offset = 2**62

    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, offset, first, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, offset + 1, shifted, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, offset, repeated, implementation_index
    )

    assert torch.equal(first, repeated)
    assert torch.equal(first[1:], shifted[:-1])
    assert not torch.equal(first, shifted)


def test_top_k_top_p_sampling_from_logits_nvidia_flat_distribution(pytestconfig):
    requested_devices = pytestconfig.getoption("--devices") or ()
    if "nvidia" not in requested_devices:
        return

    assert torch.cuda.is_available()
    assert 0 in infini.ops.TopKTopPSamplingFromLogits.active_implementation_indices(
        "nvidia"
    )

    batch_size = 4096
    vocab_size = 8
    logits = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device="cuda")
    top_k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
    top_p = torch.ones((batch_size,), dtype=torch.float64)
    out = torch.empty((batch_size,), dtype=torch.int32, device="cuda")

    _top_k_top_p_sampling_from_logits(logits, top_k, top_p, 20260811, 0, out, 0)

    counts = torch.bincount(out.to(torch.int64), minlength=vocab_size).cpu()
    expected_count = batch_size / vocab_size
    assert counts.numel() == vocab_size
    assert torch.all(torch.abs(counts - expected_count) < expected_count * 0.15)


def test_top_k_top_p_sampling_from_logits_joint_filter_semantics(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA joint-filter coverage requires CUDA")

    batch_size = 4096
    probabilities = torch.tensor(
        (0.4, 0.3, 0.2, 0.1), dtype=torch.float32, device=device
    )
    logits = probabilities.log().expand(batch_size, -1).contiguous()
    top_k = torch.full((batch_size,), 2, dtype=torch.int32)
    top_p = torch.full((batch_size,), 0.5, dtype=torch.float64)
    joint = torch.empty((batch_size,), dtype=torch.int32, device=device)
    repeated_joint = torch.empty_like(joint)
    top_k_first = torch.empty_like(joint)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        20260811,
        0,
        joint,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        20260811,
        0,
        repeated_joint,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        20260811,
        0,
        top_k_first,
        implementation_index,
    )

    assert torch.equal(joint, repeated_joint)
    assert torch.count_nonzero(top_k_first).item() == 0
    assert torch.all((joint == 0) | (joint == 1))
    second_token_count = torch.count_nonzero(joint == 1).item()
    assert batch_size * 0.15 < second_token_count < batch_size * 0.25


@pytest.mark.parametrize(
    "top_p_value, second_logit",
    (
        (1e-300, 0.0),
        (0.999999975, -17.72753356339242),
    ),
)
def test_top_k_top_p_sampling_from_logits_float64_top_p_boundaries(
    top_p_value,
    second_logit,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA float64 `top_p` coverage requires CUDA")

    logits = torch.tensor(((0.0, second_logit),), dtype=torch.float32, device=device)
    top_k = torch.tensor((2,), dtype=torch.int32)
    top_p = torch.tensor((top_p_value,), dtype=torch.float64)
    out = torch.empty((1,), dtype=torch.int32, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        5441626385717431455,
        0,
        out,
        implementation_index,
    )

    assert out.item() == 0


@pytest.mark.parametrize("indices_dtype", (torch.int32, torch.int64))
def test_top_k_top_p_sampling_from_logits_host_indices(
    indices_dtype,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA host `indices` coverage requires CUDA")

    logits = torch.tensor(
        (
            (0.0, 5.0, 1.0, 2.0),
            (0.0, 1.0, 2.0, 5.0),
            (5.0, 1.0, 2.0, 0.0),
        ),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor((2, 0), dtype=indices_dtype)
    top_k = torch.ones((2,), dtype=torch.int32)
    top_p = torch.ones((2,), dtype=torch.float64)
    out = torch.empty((2,), dtype=indices_dtype, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        0,
        out,
        implementation_index,
        indices=indices,
    )

    expected = torch.tensor((0, 1), dtype=indices_dtype, device=device)
    assert torch.equal(out, expected)


def test_top_k_top_p_sampling_from_logits_default_workspace_per_stream(
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("NVIDIA multi-stream coverage requires CUDA")

    batch_size = 8
    vocab_size = 8192
    logits_a = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)
    logits_b = -logits_a.flip(1).contiguous()
    top_k = torch.full((batch_size,), 64, dtype=torch.int32)
    top_p = torch.full((batch_size,), 0.95, dtype=torch.float64)
    baseline_a = torch.empty((batch_size,), dtype=torch.int32, device=device)
    baseline_b = torch.empty_like(baseline_a)

    _top_k_top_p_sampling_from_logits(
        logits_a, top_k, top_p, 1234, 9, baseline_a, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits_b, top_k, top_p, 1234, 9, baseline_b, implementation_index
    )
    torch.cuda.synchronize()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    stream_b.wait_stream(torch.cuda.current_stream())
    warm_a = torch.empty_like(baseline_a)
    warm_b = torch.empty_like(baseline_b)
    _top_k_top_p_sampling_from_logits(
        logits_a,
        top_k,
        top_p,
        1234,
        9,
        warm_a,
        implementation_index,
        stream=stream_a.cuda_stream,
    )
    stream_a.synchronize()
    _top_k_top_p_sampling_from_logits(
        logits_b,
        top_k,
        top_p,
        1234,
        9,
        warm_b,
        implementation_index,
        stream=stream_b.cuda_stream,
    )
    stream_b.synchronize()
    assert torch.equal(warm_a, baseline_a)
    assert torch.equal(warm_b, baseline_b)

    outputs_a = [torch.empty_like(baseline_a) for _ in range(4)]
    outputs_b = [torch.empty_like(baseline_b) for _ in range(4)]
    gate_stream = torch.cuda.Stream()
    gate_stream.wait_stream(torch.cuda.current_stream())
    gate = torch.cuda.Event()
    with torch.cuda.stream(gate_stream):
        torch.cuda._sleep(100_000_000)
        gate.record()

    stream_a.wait_event(gate)
    stream_b.wait_event(gate)

    for out_a, out_b in zip(outputs_a, outputs_b):
        _top_k_top_p_sampling_from_logits(
            logits_a,
            top_k,
            top_p,
            1234,
            9,
            out_a,
            implementation_index,
            stream=stream_a.cuda_stream,
        )
        _top_k_top_p_sampling_from_logits(
            logits_b,
            top_k,
            top_p,
            1234,
            9,
            out_b,
            implementation_index,
            stream=stream_b.cuda_stream,
        )

    stream_c = torch.cuda.Stream()
    stream_c.wait_stream(torch.cuda.current_stream())
    turnover_c = torch.empty_like(baseline_a)
    turnover_a = torch.empty_like(baseline_b)
    _top_k_top_p_sampling_from_logits(
        logits_a,
        top_k,
        top_p,
        1234,
        9,
        turnover_c,
        implementation_index,
        stream=stream_c.cuda_stream,
    )
    _top_k_top_p_sampling_from_logits(
        logits_b,
        top_k,
        top_p,
        1234,
        9,
        turnover_a,
        implementation_index,
        stream=stream_a.cuda_stream,
    )

    stream_a.synchronize()
    stream_b.synchronize()
    stream_c.synchronize()

    for out_a, out_b in zip(outputs_a, outputs_b):
        assert torch.equal(out_a, baseline_a)
        assert torch.equal(out_b, baseline_b)
    assert torch.equal(turnover_c, baseline_a)
    assert torch.equal(turnover_a, baseline_b)


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
    stream=None,
    filter_apply_order="top_k_first",
):
    if stream is None:
        stream = get_stream(logits.device)

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
        stream=stream,
        implementation_index=implementation_index,
    )
