import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


# Format:
# (input_shape, weight_shape, input_strides, weight_strides, out_strides, input_dtype)
_TEST_CASES = (
    ((1, 5), (32000, 4), None, None, None, torch.int64),
    ((2, 10), (32000, 2048), None, None, None, torch.int32),
    ((1, 5), (10, 10), None, None, None, torch.int64),
    ((2, 4), (32, 8), None, None, None, torch.int64),
    ((2, 4), (32, 8), (8, 1), None, (32, 8, 1), torch.int32),
    ((2, 4), (32, 8), None, (1, 32), None, torch.int64),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, weight_strides, out_strides, input_dtype",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-3, 0.0),
        (torch.float16, 1e-2, 0.0),
        (torch.bfloat16, 5e-2, 0.0),
    ),
)
def test_embedding(
    input_shape,
    weight_shape,
    input_strides,
    weight_strides,
    out_strides,
    input_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    vocab_size = weight_shape[0]
    embedding_dim = weight_shape[1]
    output_shape = (*input_shape, embedding_dim)

    input = randint_strided(
        1,
        min(9, vocab_size),
        input_shape,
        input_strides,
        dtype=input_dtype,
        device=device,
    )
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(output_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _embedding(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_embedding,
        (input, weight),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


def _embedding(input, weight, *, out=None, implementation_index=0):
    infini.ops.embedding(
        input,
        weight,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_embedding(input, weight, *, out=None):
    result = torch.nn.functional.embedding(input, weight)

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
