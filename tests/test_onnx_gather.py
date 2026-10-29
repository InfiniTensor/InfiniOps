from contextlib import nullcontext

import numpy as np
import pytest
import torch

import infini.ops

from tests.utils import get_stream


@pytest.mark.parametrize(
    "shape, values, axis",
    [
        ((6, 4), [[0, -1], [2, -6]], 0),
        ((2, 5, 3), [[-1, 0], [2, -5]], 1),
        ((2, 3, 4), -1, -1),
        ((7,), [[-1, 0], [2, 3]], 0),
        ((3, 4), [], 0),
        ((3, 4), [[], []], 0),
        ((0, 4), [], 0),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_onnx_gather(shape, values, axis, index_dtype, strided, device):
    x = torch.arange(int(np.prod(shape)), device=device, dtype=torch.float32)
    x = x.reshape(shape)

    if strided:
        storage = torch.empty((*shape, 2), device=device)
        storage[..., 0] = x
        x = storage[..., 0]

    index = torch.tensor(values, dtype=index_dtype, device=device)

    if strided:
        index_storage = torch.empty((*index.shape, 2), device=device, dtype=index_dtype)
        index_storage[..., 0] = index
        index = index_storage[..., 0]

    expected = np.take(x.cpu().numpy(), np.asarray(values, dtype=np.int64), axis)
    backing = torch.full((*expected.shape, 2), -123.0, device=device)
    out = backing[..., 0] if strided else torch.empty(expected.shape, device=device)
    address = out.data_ptr()
    infini.ops.onnx_gather(
        x, index, axis, out, implementation_index=8, stream=get_stream(device)
    )
    torch.testing.assert_close(out.cpu(), torch.from_numpy(np.asarray(expected)))
    assert tuple(out.shape) == expected.shape
    assert out.data_ptr() == address

    if strided:
        assert torch.all(backing[..., 1] == -123.0)


@pytest.mark.parametrize("strides", [(0, 4, 1), (0, 0, 0)])
def test_onnx_gather_empty_output_with_zero_strides(strides, device):
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    index = torch.empty((2, 0), device=device, dtype=torch.int64)
    out = torch.empty_strided((2, 0, 4), strides, device=device)
    address = out.data_ptr()

    infini.ops.onnx_gather(
        x, index, 0, out, implementation_index=8, stream=get_stream(device)
    )

    assert out.numel() == 0
    assert tuple(out.shape) == (2, 0, 4)
    assert out.stride() == strides
    assert out.data_ptr() == address


@pytest.mark.parametrize(
    "bad", ["axis", "shape", "index_dtype", "out_dtype", "broadcast"]
)
def test_onnx_gather_rejects_invalid_metadata(bad, device):
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    index = torch.tensor([0, 2], device=device, dtype=torch.int64)
    out = torch.empty((2, 4), device=device)
    axis = 2 if bad == "axis" else 0

    if bad == "shape":
        out = torch.empty((4, 2), device=device)
    elif bad == "index_dtype":
        index = index.float()
    elif bad == "out_dtype":
        out = out.to(torch.int64)
    elif bad == "broadcast":
        out = torch.empty_strided((2, 4), (0, 1), device=device)

    with pytest.raises((ValueError, RuntimeError)):
        infini.ops.onnx_gather(
            x, index, axis, out, implementation_index=8, stream=get_stream(device)
        )


@pytest.mark.parametrize("value", [-4, 3])
@pytest.mark.parametrize("device", ["cpu"])
def test_onnx_gather_rejects_out_of_range_indices(value, device):
    # A CUDA index assertion poisons the process; validate data errors on CPU.
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    index = torch.tensor([value], dtype=torch.int64, device=device)

    with pytest.raises((IndexError, RuntimeError)):
        infini.ops.onnx_gather(x, index, 0, torch.empty((1, 4)), implementation_index=8)


@pytest.mark.parametrize("partial", [False, True])
def test_onnx_gather_rejects_output_alias(device, partial):
    x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
    original = x.clone()
    index = torch.tensor(
        [2, 0] if partial else [2, 1, 0], device=device, dtype=torch.int64
    )
    out = x[1:] if partial else x
    # Populate the operator cache using independent output storage first.
    infini.ops.onnx_gather(
        x,
        index,
        0,
        torch.empty_like(out),
        implementation_index=8,
        stream=get_stream(device),
    )

    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        infini.ops.onnx_gather(
            x, index, 0, out, implementation_index=8, stream=get_stream(device)
        )

    torch.testing.assert_close(x, original)


def test_onnx_gather_rejects_output_overlapping_indices(device):
    x = torch.arange(12, device=device, dtype=torch.int64).reshape(3, 4)
    storage = torch.zeros(8, device=device, dtype=torch.int64)
    storage[:2] = torch.tensor([2, 0], device=device)

    with pytest.raises(RuntimeError, match="overlap"):
        infini.ops.onnx_gather(
            x,
            storage[:2],
            0,
            storage.view(2, 4),
            implementation_index=8,
            stream=get_stream(device),
        )


def test_onnx_gather_repeated_inputs_and_shapes(device):
    for rows in [1, 7, 3, 1]:
        for offset in [0.0, 10.0]:
            x = torch.arange(20, device=device, dtype=torch.float32).reshape(5, 4)
            x += offset
            index = torch.arange(rows, device=device, dtype=torch.int64) % 5
            out = torch.empty((rows, 4), device=device)
            infini.ops.onnx_gather(
                x, index, 0, out, implementation_index=8, stream=get_stream(device)
            )
            torch.testing.assert_close(out, x[index])


def test_onnx_gather_rejects_internally_overlapping_output(device):
    x = torch.arange(9, device=device, dtype=torch.float32).reshape(3, 3)
    index = torch.tensor([0, 2], device=device, dtype=torch.int64)
    out = torch.empty(4, device=device).as_strided((2, 3), (1, 1))

    with pytest.raises(RuntimeError, match="overlap"):
        infini.ops.onnx_gather(
            x, index, 0, out, implementation_index=8, stream=get_stream(device)
        )


def test_onnx_gather_nondefault_stream(device):
    accelerator = getattr(torch, torch.device(device).type) if device != "cpu" else None
    stream = accelerator.Stream() if accelerator is not None else None

    with accelerator.stream(stream) if stream is not None else nullcontext():
        x = torch.arange(20, device=device, dtype=torch.float32).reshape(5, 4)
        index = torch.tensor([4, 0], device=device, dtype=torch.int64)
        out = torch.empty((2, 4), device=device)

    original = get_stream(device)
    infini.ops.onnx_gather(
        x,
        index,
        0,
        out,
        implementation_index=8,
        stream=getattr(stream, f"{torch.device(device).type}_stream")
        if stream is not None
        else original,
    )
    assert get_stream(device) == original
    if stream is not None:
        stream.synchronize()

    torch.testing.assert_close(out, x[index])
