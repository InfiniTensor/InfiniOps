import infini.ops
import pytest
import torch

from tests.utils import empty_strided, randn_strided

# Only CPU implementation exists for now.
_CPU_ONLY = pytest.mark.parametrize("device", ("cpu",))


# --- Helpers ---


def _random_sample(
    logits,
    out,
    valid,
    temperature=None,
    temperature_val=1.0,
    top_k=None,
    top_k_val=0,
    top_p=None,
    top_p_val=1.0,
    min_p=None,
    min_p_val=0.0,
    seed=0,
    offset=0,
    deterministic=True,
):
    infini.ops.random_sample(
        logits,
        out,
        valid,
        temperature,
        temperature_val,
        top_k,
        top_k_val,
        top_p,
        top_p_val,
        min_p,
        min_p_val,
        seed,
        offset,
        deterministic,
    )
    return out, valid


def _torch_argmax_sample(logits):
    return torch.argmax(logits, dim=-1)


# --- Tests ---


@pytest.mark.parametrize("batch_size, vocab_size", ((1, 16), (4, 128), (8, 256)))
@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_greedy_topk1(batch_size, vocab_size, dtype, device):
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)
    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, top_k_val=1, seed=42)

    expected = _torch_argmax_sample(logits)
    assert torch.equal(out, expected), (
        f"top_k=1 should give argmax, got {out}, expected {expected}"
    )
    assert valid.all(), "all samples should be valid"


@pytest.mark.parametrize("batch_size, vocab_size", ((1, 16), (4, 64)))
@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_reproducibility(batch_size, vocab_size, dtype, device):
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)

    out1 = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid1 = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)
    out2 = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid2 = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out1, valid1, seed=123, offset=0, deterministic=True)
    _random_sample(logits, out2, valid2, seed=123, offset=0, deterministic=True)

    assert torch.equal(out1, out2), "same seed should give same output"
    assert torch.equal(valid1, valid2)


@pytest.mark.parametrize("batch_size, vocab_size", ((2, 32), (4, 64)))
@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_output_valid(batch_size, vocab_size, dtype, device):
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)
    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, seed=42)

    assert valid.all(), "all samples should be valid for normal inputs"
    assert (out >= 0).all() and (out < vocab_size).all(), "sampled indices out of range"


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_topp_filtering(dtype, device):
    batch_size, vocab_size = 4, 16
    logits = torch.full((batch_size, vocab_size), -10.0, dtype=dtype, device=device)
    logits[:, 0] = 10.0

    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, top_p_val=0.5, seed=42)

    assert (out == 0).all(), "top_p=0.5 should always pick the dominant token"


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_minp_filtering(dtype, device):
    batch_size, vocab_size = 4, 16
    logits = torch.full((batch_size, vocab_size), -10.0, dtype=dtype, device=device)
    logits[:, 3] = 10.0

    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, min_p_val=0.5, seed=42)

    assert (out == 3).all(), "min_p=0.5 should always pick the dominant token"


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_1d_logits(dtype, device):
    vocab_size = 32
    logits = randn_strided((vocab_size,), None, dtype=dtype, device=device)
    out = empty_strided((1,), None, dtype=torch.int32, device=device)
    valid = empty_strided((1,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, top_k_val=1, seed=42)

    expected = _torch_argmax_sample(logits.unsqueeze(0))
    assert torch.equal(out, expected)
    assert valid.all()


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_seed_offset_reproducibility(dtype, device):
    """Same seed+offset reproduces; different seed likely differs."""
    batch_size, vocab_size = 4, 256
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)

    out1 = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid1 = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)
    out2 = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid2 = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)
    out3 = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid3 = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    # Same seed + offset → must be identical
    _random_sample(logits, out1, valid1, seed=1, offset=0)
    _random_sample(logits, out2, valid2, seed=1, offset=0)
    assert torch.equal(out1, out2), "same seed+offset should reproduce"

    # Different offset → must be different (different RNG state)
    _random_sample(logits, out3, valid3, seed=1, offset=999999)
    assert not torch.equal(out1, out3), (
        "different offset should produce different results"
    )


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_int64_output(dtype, device):
    batch_size, vocab_size = 2, 32
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)
    out = empty_strided((batch_size,), None, dtype=torch.int64, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    _random_sample(logits, out, valid, top_k_val=1, seed=42)

    expected = _torch_argmax_sample(logits)
    assert out.dtype == torch.int64
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_per_batch_tensor_params(dtype, device):
    """Per-batch tensor parameters (int64 top_k, float32 temperature) should work."""
    batch_size, vocab_size = 4, 32
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)
    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    # top_k as int64 tensor: batch 0 uses top_k=1 (greedy), others use top_k=0 (no filter).
    top_k_tensor = torch.tensor([1, 0, 0, 0], dtype=torch.int64, device=device)

    _random_sample(logits, out, valid, top_k=top_k_tensor, seed=42)

    # Batch 0 must be argmax (top_k=1).
    assert out[0].item() == torch.argmax(logits[0]).item()
    assert valid.all()


@pytest.mark.parametrize("dtype", (torch.float32,))
@_CPU_ONLY
def test_per_batch_temperature_tensor(dtype, device):
    """Per-batch float32 temperature tensor should work."""
    batch_size, vocab_size = 4, 32
    logits = randn_strided((batch_size, vocab_size), None, dtype=dtype, device=device)
    out = empty_strided((batch_size,), None, dtype=torch.int32, device=device)
    valid = empty_strided((batch_size,), None, dtype=torch.uint8, device=device)

    # Very low temperature → near-deterministic for all batches.
    temp_tensor = torch.full((batch_size,), 0.01, dtype=torch.float32, device=device)

    _random_sample(logits, out, valid, temperature=temp_tensor, seed=42)

    expected = _torch_argmax_sample(logits)
    assert torch.equal(out, expected), "near-zero temperature should give argmax"
    assert valid.all()
