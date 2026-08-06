import pytest
import torch

import infini.ops


class _IteratingTuple(tuple):
    def __new__(cls, stored_values, iterated_values):
        instance = super().__new__(cls, stored_values)
        instance.iterated_values = tuple(iterated_values)

        return instance

    def __iter__(self):
        return iter(self.iterated_values)


class _DuckTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = _IteratingTuple((2, 2), tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device

    def data_ptr(self):
        return self.tensor.data_ptr()

    def stride(self):
        return _IteratingTuple((2, 1), self.tensor.stride())


def test_duck_tensor_conversion_respects_sequence_subclass_iteration():
    implementation_indices = infini.ops.Add.active_implementation_indices("cpu")

    if not implementation_indices:
        pytest.skip("CPU Add implementation is not active")

    input = torch.tensor(
        (
            (-1.0, -2.0, -3.0),
            (-4.0, -5.0, -6.0),
        )
    )
    other = torch.tensor(
        (
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        )
    )
    out = torch.full_like(input, -123.0)

    infini.ops.add(
        _DuckTensor(input),
        _DuckTensor(other),
        _DuckTensor(out),
        stream=0,
        implementation_index=implementation_indices[0],
    )

    assert torch.equal(out, input + other)
