import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import get_stream


@pytest.mark.operator_set({"MaxPool2d": 8})
@pytest.mark.parametrize("shape", [(1, 2, 7, 9), (2, 3, 8, 10), (2, 7, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel,stride,padding,dilation,ceil",
    [
        ([1, 1], [1, 1], [0, 0], [1, 1], False),
        ([2, 3], [2, 2], [0, 1], [1, 1], False),
        ([2, 2], [2, 2], [0, 0], [1, 1], True),
    ],
)
def test_value_only_pool(shape, dtype, kernel, stride, padding, dilation, ceil, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    expected = F.max_pool2d(x, kernel, stride, padding, dilation, ceil)
    out = torch.empty_like(expected)
    infini.ops.max_pool2d(
        x,
        kernel,
        stride,
        padding,
        dilation,
        ceil,
        out,
        implementation_index=8,
        stream=get_stream(x.device),
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
