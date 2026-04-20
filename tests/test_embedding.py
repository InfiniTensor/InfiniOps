import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randint_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, vocab_size, embedding_dim",
    (
        ((7,), 128, 32),
        ((2, 5), 256, 64),
        ((1, 1024), 73448, 4096),
    ),
)
@pytest.mark.parametrize(
    ("weight_dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_embedding(
    input_shape,
    vocab_size,
    embedding_dim,
    weight_dtype,
    index_dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.Embedding.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    input = randint_strided(
        0, vocab_size, input_shape, None, dtype=index_dtype, device=device
    )
    weight = randn_strided(
        (vocab_size, embedding_dim), None, dtype=weight_dtype, device=device
    )
    out = empty_strided(
        input_shape + (embedding_dim,), None, dtype=weight_dtype, device=device
    )

    return Payload(
        _embedding,
        _torch_embedding,
        (input, weight, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _embedding(input, weight, out):
    infini.ops.embedding(input, weight, out, implementation_index=1)

    return out


def _torch_embedding(input, weight, out):
    out.copy_(torch.nn.functional.embedding(input, weight))

    return out
