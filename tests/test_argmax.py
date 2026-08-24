import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
)
def test_argmax_flattened(dtype, device, implementation_index):
    if device != "cuda":
        pytest.skip("argmax requires the NVIDIA backend")

    input = torch.randn(32_003, dtype=dtype, device=device)
    input[17_391] = 100
    out = torch.full((), -1, dtype=torch.int64, device=device)

    result = infini.ops.argmax(
        input,
        None,
        False,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    assert out.item() == torch.argmax(input).item()
