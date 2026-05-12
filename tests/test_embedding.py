import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, vocab_size, hidden_size",
    (
        ((5,), 17, 8),
        ((2, 3), 23, 16),
    ),
)
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0.0, 0.0),
        (torch.float16, 0.0, 0.0),
        (torch.bfloat16, 0.0, 0.0),
    ),
)
def test_embedding(
    input_shape,
    vocab_size,
    hidden_size,
    index_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input_ids = torch.randint(
        0, vocab_size, input_shape, dtype=index_dtype, device=device
    )
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype, device=device)
    out = torch.empty((*input_shape, hidden_size), dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _embedding(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_embedding,
        (input_ids, weight, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _embedding(input_ids, weight, out, *, implementation_index=0):
    infini.ops.embedding(
        input_ids,
        weight,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input_ids.device),
    )

    return out


def _ref_embedding(input_ids, weight, out):
    del out

    return torch.nn.functional.embedding(input_ids.long(), weight)
