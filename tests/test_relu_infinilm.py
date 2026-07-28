import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((), None, None, False),
        ((0,), None, None, False),
        ((1, 3), None, None, False),
        ((1, 3), None, None, True),
        ((3, 3), None, None, False),
        ((3, 3), (5, 1), (5, 1), False),
        ((32, 20, 512), None, None, False),
        ((32, 20, 512), None, None, True),
        ((33, 333, 333), None, None, False),
        ((32, 256, 112, 112), None, None, False),
        ((3, 3, 13, 9, 17), None, None, False),
        (
            (3, 3, 13, 9, 17),
            (19890, 6630, 510, 34, 1),
            (19890, 6630, 510, 34, 1),
            False,
        ),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 1e-15, 1e-15),
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-3, 1e-3),
    ),
)
def test_relu_infinilm(
    shape, input_strides, out_strides, inplace, dtype, device, rtol, atol
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 ReLU_INFINILM")

    input = rand_strided(shape, input_strides, dtype=dtype, device=device)
    input.mul_(2).sub_(1)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        _relu_infinilm,
        _torch_relu_infinilm,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize(
    "dtype",
    (torch.float64, torch.float32, torch.float16, torch.bfloat16),
)
def test_relu_infinilm_matches_special_value_semantics(dtype, inplace, device):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 ReLU_INFINILM")

    input = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=dtype,
        device=device,
    )
    expected = torch.nn.functional.relu(input).cpu()
    out = input if inplace else torch.empty_like(input)

    actual = _relu_infinilm(input, out).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zero_mask = expected == 0
    assert torch.equal(
        torch.signbit(actual[zero_mask]),
        torch.signbit(expected[zero_mask]),
    )


def _relu_infinilm(input, out):
    infini.ops.relu_infinilm(
        input,
        out,
        stream=get_stream(input.device),
        implementation_index=0,
    )

    return out


def _torch_relu_infinilm(input, out):
    result = torch.nn.functional.relu(input)
    out.copy_(result)

    return out
