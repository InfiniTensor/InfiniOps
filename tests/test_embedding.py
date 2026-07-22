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
# (input_shape, weight_shape, input_strides, weight_strides, out_strides,
#  input_dtype, options)
_TEST_CASES = tuple(
    (*case, None)
    for case in (
        ((1, 5), (32000, 4), None, None, None, torch.int64),
        ((2, 10), (32000, 2048), None, None, None, torch.int32),
        ((1, 5), (10, 10), None, None, None, torch.int64),
        ((2, 4), (32, 8), None, None, None, torch.int64),
        ((2, 4), (32, 8), (8, 1), None, (32, 8, 1), torch.int32),
        ((2, 4), (32, 8), None, (1, 32), None, torch.int64),
    )
) + tuple(
    ((2, 3), (8, 4), None, None, None, torch.int64, options)
    for options in (
        (-1, False, False),
        (-1, False, True),
        (-1, True, False),
        (-1, True, True),
        (0, False, False),
        (0, False, True),
        (0, True, False),
        (0, True, True),
    )
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "weight_shape",
        "input_strides",
        "weight_strides",
        "out_strides",
        "input_dtype",
        "options",
    ),
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
    options,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    vocab_size = weight_shape[0]
    embedding_dim = weight_shape[1]
    output_shape = (*input_shape, embedding_dim)
    padding_idx, scale_grad_by_freq, sparse = options or (None, False, False)

    input = randint_strided(
        0 if padding_idx is not None else 1,
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
            *args,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            implementation_index=implementation_index,
            **kwargs,
        ),
        lambda *args, **kwargs: _torch_embedding(
            *args,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            **kwargs,
        ),
        (weight, input),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


def _embedding(
    weight,
    indices,
    *,
    out,
    padding_idx,
    scale_grad_by_freq,
    sparse,
    implementation_index,
):
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(indices.device),
    }

    if padding_idx is None:
        infini.ops.embedding(weight, indices, out, **kwargs)
    else:
        infini.ops.embedding(
            weight,
            indices,
            padding_idx,
            scale_grad_by_freq,
            sparse,
            out,
            **kwargs,
        )

    return out


def _torch_embedding(
    weight,
    indices,
    *,
    out,
    padding_idx,
    scale_grad_by_freq,
    sparse,
):
    result = torch.nn.functional.embedding(
        indices,
        weight,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )
    out.copy_(result)

    return out
