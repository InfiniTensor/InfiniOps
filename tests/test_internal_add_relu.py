import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("other_shape", [(2, 3, 5, 7), (1,), (1, 3, 1, 1)])
def test_iluvatar_add_relu_fp32_values(device, other_shape):
    if not infini.ops.InternalAddRelu.active_implementation_indices("iluvatar"):
        pytest.skip("Iluvatar Add/ReLU is not enabled")
    if torch.device(device).type == "cpu":
        pytest.skip("Iluvatar Add/ReLU requires the device backend")
    values = torch.tensor(
        [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")]
    )
    source = values.repeat(30).reshape(2, 3, 5, 7).to(device)
    other = torch.zeros(other_shape, dtype=torch.float32, device=device)
    out = torch.empty_like(source)
    infini.ops.internal_add_relu(
        source,
        other,
        1.0,
        out,
        stream=get_stream(source.device),
        implementation_index=0,
    )
    expected = torch.relu(source.cpu() + other.cpu())
    actual = out.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(
        actual[valid].view(torch.int32), expected[valid].view(torch.int32)
    )
