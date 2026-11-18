import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


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
@pytest.mark.parametrize("strided_upstream", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_slice_backward(
    shape,
    strides,
    dim,
    start,
    end,
    step,
    strided_upstream,
    dtype,
    device,
    implementation_index,
):
    grad_input = empty_strided(shape, strides, dtype=dtype, device=device)
    index = [slice(None)] * len(shape)
    index[dim] = slice(start, end, step)
    output_shape = grad_input[tuple(index)].shape
    output_strides = (
        tuple(stride * 2 for stride in torch.empty(output_shape).stride())
        if strided_upstream
        else None
    )
    grad_output = randn_strided(
        output_shape, output_strides, dtype=dtype, device=device
    )
    grad_input.fill_(17)

    def actual(grad_output, grad_input):
        infini.ops.slice_backward(
            grad_output,
            shape,
            dim,
            start,
            end,
            step,
            grad_input,
            stream=get_stream(device),
            implementation_index=implementation_index,
        )
        return grad_input

    def reference(grad_output, grad_input):
        x = torch.zeros(shape, dtype=dtype, device=device, requires_grad=True)
        with torch.enable_grad():
            result = x[tuple(index)]
            gradient = torch.autograd.grad(result, x, grad_output)[0]
        grad_input.copy_(gradient)
        return grad_input

    return Payload(actual, reference, (grad_output, grad_input), {}, rtol=0, atol=0)


@pytest.mark.parametrize("offset", (0, 2))
def test_slice_backward_alias(offset, device, implementation_index):
    backing = torch.arange(32, dtype=torch.float32, device=device)
    grad_output = backing[offset : offset + 6].reshape(3, 2)
    out = backing[:12].reshape(3, 4)
    expected = torch.zeros_like(out)
    expected[:, ::2] = grad_output.clone()
    infini.ops.slice_backward(
        grad_output,
        (3, 4),
        1,
        0,
        4,
        2,
        out,
        stream=get_stream(device),
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
