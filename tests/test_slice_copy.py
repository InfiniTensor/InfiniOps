import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


_CASES = (
    ((17,), None, 0, 1, 16, 3),
    ((6, 9), (12, 1), 0, 1, 6, 2),
    ((6, 9), (12, 1), 1, 0, 9, 3),
    ((4, 6, 7), (60, 9, 1), -2, -5, 6, 2),
    ((2, 3, 4, 5), None, 3, 0, 5, 2),
    ((5, 7), None, 0, 3, 3, 1),
    ((3, 5), None, -1, -100, 100, 2),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape,strides,dim,start,end,step", _CASES)
@pytest.mark.parametrize("optional_bounds", (False, True))
@pytest.mark.parametrize("strided_output", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_slice_copy(
    shape,
    strides,
    dim,
    start,
    end,
    step,
    optional_bounds,
    strided_output,
    dtype,
    device,
    implementation_index,
):
    input = randn_strided(shape, strides, dtype=dtype, device=device)
    start, end = (None, None) if optional_bounds else (start, end)
    index = [slice(None)] * len(shape)
    index[dim] = slice(start, end, step)
    expected_shape = input[tuple(index)].shape
    out_strides = tuple(
        stride * (2 if strided_output else 1)
        for stride in torch.empty(expected_shape).stride()
    )
    out = torch.empty_strided(expected_shape, out_strides, dtype=dtype, device=device)

    def actual(input, out):
        infini.ops.slice_copy(
            input,
            dim,
            start,
            end,
            step,
            out,
            stream=get_stream(device),
            implementation_index=implementation_index,
        )
        return out

    def reference(input, out):
        out.copy_(input[tuple(index)].clone())
        return out

    return Payload(actual, reference, (input, out), {}, rtol=0, atol=0)


@pytest.mark.parametrize("offset", (0, 2))
def test_slice_copy_alias(offset, device, implementation_index):
    backing = torch.arange(32, dtype=torch.float32, device=device)
    input = backing[:12].reshape(3, 4)
    out = backing[offset : offset + 6].reshape(3, 2)
    expected = input[:, ::2].clone()
    infini.ops.slice_copy(
        input,
        1,
        None,
        None,
        2,
        out,
        stream=get_stream(device),
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
