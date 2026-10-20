"""ONNX Gather semantics must not be confused with torch.gather."""

import pytest
import torch

import infini.ops

from tests.utils import get_stream


@pytest.mark.parametrize(
    "shape,index_shape,axis",
    [
        ((7,), (), 0),
        ((7,), (2, 3), -1),
        ((3, 7, 5), (2, 3), 1),
        ((3, 7, 5), (2,), -2),
        ((3, 7, 5), (0, 2), 1),
        ((2, 3, 4), (2, 1), 0),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_onnx_gather_multirank_negative_and_empty_indices(
    device, shape, index_shape, axis, index_dtype, strided
):
    x = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float32, device=device
    ).reshape(shape)
    if strided:
        backing = torch.empty((*shape, 2), device=device)
        backing[..., 0].copy_(x)
        x = backing[..., 0]
    dim = axis % len(shape)
    count = torch.tensor(index_shape).prod().item() if index_shape else 1
    indices = torch.arange(count, device=device, dtype=index_dtype) % shape[dim]
    indices[::2] -= shape[dim]
    indices = indices.reshape(index_shape)
    expected = torch.index_select(x, dim, (indices % shape[dim]).reshape(-1).long())
    output_shape = (*shape[:dim], *index_shape, *shape[dim + 1 :])
    expected = expected.reshape(output_shape)
    storage = torch.full((*output_shape, 2), -123.0, device=device)
    out = storage[..., 0] if strided else torch.empty(output_shape, device=device)
    address = out.data_ptr()
    infini.ops.onnx_gather(
        x, indices, axis, out, stream=get_stream(x.device), implementation_index=8
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.data_ptr() == address
    if strided:
        assert torch.all(storage[..., 1] == -123.0)


def test_onnx_gather_rejects_invalid_metadata_before_launch(device):
    x = torch.ones((2, 3), device=device)
    indices = torch.zeros((2,), dtype=torch.int64, device=device)
    out = torch.empty((2, 2), device=device)
    with pytest.raises((ValueError, RuntimeError)):
        infini.ops.onnx_gather(
            x, indices, 2, out, implementation_index=8, stream=get_stream(x.device)
        )
    with pytest.raises((ValueError, RuntimeError)):
        infini.ops.onnx_gather(
            x,
            indices,
            0,
            torch.empty((1,), device=device),
            implementation_index=8,
            stream=get_stream(x.device),
        )


def test_onnx_gather_rechecks_output_alias_after_cache_hit(device):
    x = torch.arange(7, dtype=torch.float32, device=device)
    indices = torch.arange(6, -1, -1, dtype=torch.int64, device=device)
    infini.ops.onnx_gather(
        x,
        indices,
        0,
        torch.empty_like(x),
        implementation_index=8,
        stream=get_stream(x.device),
    )
    before = x.clone()
    with pytest.raises((ValueError, RuntimeError)):
        infini.ops.onnx_gather(
            x, indices, 0, x, implementation_index=8, stream=get_stream(x.device)
        )
    torch.testing.assert_close(x, before)
