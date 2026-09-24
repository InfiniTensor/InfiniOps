import infini.ops
import pytest
import torch

from tests.utils import empty_strided, get_stream, randn_strided


@pytest.mark.parametrize(
    "input_shape, weight_shape",
    (
        ((1, 16), (16, 16)),
        ((3, 32), (32, 48)),
        ((17, 64), (64, 80)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_trans_matmul_weight_nz(
    input_shape,
    weight_shape,
    dtype,
    device,
    rtol,
    atol,
):
    if device != "npu":
        pytest.skip("ACLNN MatmulWeightNz is an Ascend-only path")

    input = randn_strided(input_shape, None, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, None, dtype=dtype, device=device)
    converted = empty_strided(weight.shape, None, dtype=dtype, device=device)
    output = empty_strided(
        (input.shape[0], weight.shape[1]), None, dtype=dtype, device=device
    )
    stream = get_stream(input.device)

    infini.ops.trans_matmul_weight_nz(weight, converted, stream=stream)
    infini.ops.matmul_weight_nz(input, converted, output, stream=stream)

    reference = torch.matmul(input.float(), weight.float()).to(dtype)
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
