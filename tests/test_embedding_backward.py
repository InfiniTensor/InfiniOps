import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format:
# (indices, num_weights, embedding_dim, indices_strides,
#  grad_output_strides, grad_weight_strides, padding_idx,
#  scale_grad_by_freq)
_TEST_CASES = (
    pytest.param(
        (1, 2, 1, 4, 2, 1),
        6,
        4,
        None,
        None,
        None,
        -1,
        False,
        id="repeated-indices",
    ),
    pytest.param(
        ((0, 1, 0), (2, 1, 3)),
        5,
        7,
        (5, 1),
        (32, 8, 1),
        (9, 1),
        1,
        False,
        id="padding-and-strides",
    ),
    pytest.param(
        (2, 2, 2, 3, 3, 4),
        7,
        5,
        None,
        None,
        None,
        -1,
        True,
        id="scale-grad-by-frequency",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "indices_values",
        "num_weights",
        "embedding_dim",
        "indices_strides",
        "grad_output_strides",
        "grad_weight_strides",
        "padding_idx",
        "scale_grad_by_freq",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize("indices_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ),
)
def test_embedding_backward(
    indices_values,
    num_weights,
    embedding_dim,
    indices_strides,
    grad_output_strides,
    grad_weight_strides,
    padding_idx,
    scale_grad_by_freq,
    indices_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    indices_values = torch.tensor(indices_values, dtype=indices_dtype)
    indices = empty_strided(
        indices_values.shape,
        indices_strides,
        dtype=indices_dtype,
        device=device,
    )
    indices.copy_(indices_values.to(device))

    grad_output_shape = (*indices.shape, embedding_dim)
    grad_output = randn_strided(
        grad_output_shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_weight = empty_strided(
        (num_weights, embedding_dim),
        grad_weight_strides,
        dtype=dtype,
        device=device,
    )
    grad_weight.fill_(float("nan"))

    return Payload(
        lambda *args: _embedding_backward(
            *args,
            num_weights=num_weights,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_embedding_backward(
            *args,
            num_weights=num_weights,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        ),
        (grad_output, indices, grad_weight),
        {},
        rtol=rtol,
        atol=atol,
    )


def _embedding_backward(
    grad_output,
    indices,
    grad_weight,
    *,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
    implementation_index,
):
    infini.ops.embedding_backward(
        grad_output,
        indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        False,
        grad_weight,
        implementation_index=implementation_index,
        stream=get_stream(grad_output.device),
    )

    return grad_weight


def _torch_embedding_backward(
    grad_output,
    indices,
    grad_weight,
    *,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
):
    result = torch.ops.aten.embedding_backward.default(
        grad_output,
        indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        False,
    )
    grad_weight.copy_(result)

    return grad_weight
