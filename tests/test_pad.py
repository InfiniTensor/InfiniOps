import infini.ops
import pytest
import torch
import torch.nn.functional as functional

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "shape, padding, input_strides, out_strides, value",
    (
        ((2, 3, 5), (0, 3), None, None, 0.0),
        ((2, 3), (2, 1), None, None, 2.5),
        ((2, 3), (1, 2, 2, 1), (1, 2), (1, 7), -1.25),
        ((2, 3, 5), (1, 2), (0, 0, 1), None, 0.0),
        ((0, 3, 4), (1, 2), None, None, 0.0),
        ((2, 3), (0, 0), None, None, 0.0),
    ),
)
def test_pad(shape, padding, input_strides, out_strides, value, dtype, device):
    if device == "cpu" or not hasattr(infini.ops, "pad"):
        pytest.skip("No Pad provider is enabled in this build")
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    output_shape = list(shape)
    for pair in range(len(padding) // 2):
        output_shape[-1 - pair] += padding[2 * pair] + padding[2 * pair + 1]
    out = empty_strided(output_shape, out_strides, dtype=dtype, device=device)
    return Payload(_pad, _torch_pad, (input, padding, value, out), {}, rtol=0, atol=0)


def _pad(input, padding, value, out):
    infini.ops.pad(
        input, padding, "constant", value, out, stream=get_stream(input.device)
    )
    return out


def _torch_pad(input, padding, value, out):
    out.copy_(functional.pad(input, padding, "constant", value))
    return out
