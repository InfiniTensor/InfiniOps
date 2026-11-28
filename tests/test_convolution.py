import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


_TEST_CASES = (
    pytest.param((1, 2, 7), (4, 2, 3), 1, id="1d"),
    pytest.param((1, 2, 5, 6), (3, 2, 3, 3), 1, id="2d"),
    pytest.param((1, 2, 4, 5, 6), (3, 2, 3, 3, 3), 1, id="3d"),
    pytest.param((1, 4, 5, 6), (4, 2, 3, 3), 2, id="2d-groups"),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("input_shape, weight_shape, groups", _TEST_CASES)
@pytest.mark.parametrize("has_bias", (False, True))
def test_convolution(input_shape, weight_shape, groups, has_bias, device):
    input = torch.randn(input_shape, device=device) * 0.01
    weight = torch.randn(weight_shape, device=device) * 0.01
    bias = torch.randn(weight_shape[0], device=device) * 0.01 if has_bias else None
    spatial_ndim = len(input_shape) - 2
    stride = (1,) * spatial_ndim
    padding = (1,) * spatial_ndim
    dilation = (1,) * spatial_ndim
    output_padding = (0,) * spatial_ndim
    expected = torch.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        groups,
    )
    out = torch.empty_like(expected)

    return Payload(
        lambda input, weight, bias, out: _convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            out,
        ),
        lambda input, weight, bias, out: torch.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            output_padding,
            groups,
        ),
        (input, weight, bias, out),
        {},
    )


def _convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    output_padding,
    groups,
    out,
):
    infini.ops.convolution(
        input,
        weight,
        bias,
        list(stride),
        list(padding),
        list(dilation),
        False,
        list(output_padding),
        groups,
        out,
        stream=get_stream(input.device),
    )

    return out


@pytest.mark.parametrize("implementation_index", [0, 8])
@pytest.mark.parametrize(
    "shape,kernel,groups,dilation,has_bias",
    [
        ((1, 8, 7, 7), (16, 8, 1, 1), 1, 1, False),
        ((2, 8, 13, 11), (12, 8, 3, 3), 1, 1, False),
        ((1, 8, 9, 9), (12, 4, 3, 3), 2, 1, False),
        ((1, 8, 9, 9), (12, 8, 3, 3), 1, 2, True),
    ],
)
def test_iluvatar_convolution_output_storage(
    device, implementation_index, shape, kernel, groups, dilation, has_bias
):
    if implementation_index not in infini.ops.Convolution.active_implementation_indices(
        "iluvatar"
    ):
        pytest.skip("Requested Iluvatar implementation is not enabled")
    if torch.device(device).type == "cpu":
        pytest.skip("Iluvatar test requires the accelerator backend")
    generator = torch.Generator().manual_seed(20261115)
    x_host = torch.randn(shape, generator=generator) * 0.01
    w_host = torch.randn(kernel, generator=generator) * 0.01
    bias_host = torch.randn(kernel[0], generator=generator) * 0.01 if has_bias else None
    x, weight = x_host.to(device), w_host.to(device)
    bias = bias_host.to(device) if has_bias else None
    expected = torch.nn.functional.conv2d(
        x_host, w_host, bias_host, padding=1, dilation=dilation, groups=groups
    )
    out = torch.full(expected.shape, float("nan"), device=device)
    address = out.data_ptr()
    infini.ops.convolution(
        x,
        weight,
        bias,
        [1, 1],
        [1, 1],
        [dilation, dilation],
        False,
        [0, 0],
        groups,
        out,
        stream=get_stream(device),
        implementation_index=implementation_index,
    )
    assert out.data_ptr() == address
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-6)
